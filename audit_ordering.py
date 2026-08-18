#!/usr/bin/env python3
"""Auditer la qualite d'ordonnancement d'une playlist produite par le sorter.

Le sorter mesure deja deux choses a la fin d'un run (chevauchement adjacent,
familles fragmentees), mais ces deux metriques sont saturees en mode
``two_level`` : elles disent que le tri INTERNE a chaque famille est bon, pas
que l'ORDRE DES FAMILLES entre elles a du sens. Ce script mesure ce qui manque.

    python audit_ordering.py subsonic_starred_tracks_sorted_2026-08-18.csv
    python audit_ordering.py *.csv --json rapport.json

Metriques produites :

* ``cohesion``      chevauchement de Jaccard moyen entre titres voisins (micro).
* ``intrus``        titres sans aucun genre commun avec leurs DEUX voisins alors
                    que ces voisins en partagent entre eux. Un vrai intrus, par
                    opposition a une simple frontiere de bloc.
* ``macro``         part de la proximite ATTEIGNABLE reellement exploitee entre
                    familles voisines : pour chaque frontiere, similarite cosinus
                    des centroides de tags obtenue / meilleure disponible. C'est
                    la metrique qui manquait ; elle plafonne autour de 0,29.
* ``queue``         titres relegues en fin de playlist par ordre alphabetique
                    (familles a un seul album, poussees dans la queue par
                    ``_two_level_order``). Ils ne sont pas tries du tout.
* ``compilations``  titres dont l'album reunit plusieurs artistes : le genre y
                    est resolu par album, donc faux pour la plupart des pistes.
"""

import argparse
import json
import sys
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tag_set(value):
    if not isinstance(value, str) or not value.strip():
        return set()
    return {t.strip().lower() for t in value.split(",") if t.strip()}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def contiguous_blocks(seq):
    """[(valeur, index_debut, longueur)] des plages contigues."""
    out, cur, start = [], seq[0], 0
    for i in range(1, len(seq)):
        if seq[i] != cur:
            out.append((cur, start, i - start))
            cur, start = seq[i], i
    out.append((cur, start, len(seq) - start))
    return out


def audit(path):
    d = pd.read_csv(path).sort_values("Sort Order").reset_index(drop=True)
    n = len(d)
    tags = [tag_set(x) for x in d["Album Genre"]]
    has_root = "Root Genre" in d.columns
    roots = [str(x) if isinstance(x, str) and x.strip() else "Unknown"
             for x in (d["Root Genre"] if has_root else ["?"] * n)]
    js = [jaccard(tags[i], tags[i + 1]) for i in range(n - 1)]

    # --- decomposition intra / inter-album ------------------------------------
    # Les paires internes a un album partagent la meme etiquette par
    # construction (Jaccard ~1 gratuit) et gonflent la moyenne brute (~57 %
    # des paires). Seul l'inter-album mesure ce que le tri decide vraiment.
    if "Unique Album" in d.columns:
        units = [str(x) for x in d["Unique Album"]]
        intra = [js[i] for i in range(n - 1) if units[i] == units[i + 1]]
        inter = [js[i] for i in range(n - 1) if units[i] != units[i + 1]]
    else:
        intra, inter = [], js
    coh_intra = round(float(np.mean(intra)), 4) if intra else None
    coh_inter = round(float(np.mean(inter)), 4) if inter else None

    # --- intrus : casse une sequence par ailleurs coherente -------------------
    intrus = []
    for i in range(1, n - 1):
        if not (tags[i] and tags[i - 1] and tags[i + 1]):
            continue
        if (jaccard(tags[i], tags[i - 1]) == 0 and jaccard(tags[i], tags[i + 1]) == 0
                and jaccard(tags[i - 1], tags[i + 1]) > 0):
            intrus.append({
                "position": i + 1, "titre": str(d.loc[i, "Song"]),
                "artiste": str(d.loc[i, "Artist"]), "genre": str(d.loc[i, "Album Genre"]),
                "source": str(d.loc[i, "source"]) if "source" in d.columns else "",
                "precedent": f'{d.loc[i-1, "Artist"]} · {d.loc[i-1, "Album Genre"]}',
                "suivant": f'{d.loc[i+1, "Artist"]} · {d.loc[i+1, "Album Genre"]}',
            })

    blocks = contiguous_blocks(roots)
    families = Counter(v for v, _, _ in blocks)

    # --- macro : proximite obtenue vs proximite disponible --------------------
    docs = d["Album Genre"].fillna("").astype(str).str.lower().str.replace(",", " ", regex=False)
    V = TfidfVectorizer(token_pattern=r"[^\s]+").fit_transform(docs)
    members = {}
    for v, s, l in blocks:
        members.setdefault(v, []).extend(range(s, s + l))
    order = [v for v, _, _ in blocks]
    C = np.vstack([np.asarray(V[members[v]].mean(axis=0)) for v in order])
    S = cosine_similarity(C)
    transitions = []
    for i in range(len(order) - 1):
        row = S[i].copy()
        row[i] = -1
        transitions.append({
            "de": order[i], "vers": order[i + 1],
            "obtenu": round(float(S[i, i + 1]), 3),
            "meilleur_disponible": round(float(row.max()), 3),
            "position": blocks[i + 1][1],
        })
    got = np.mean([t["obtenu"] for t in transitions]) if transitions else 0.0
    best = np.mean([t["meilleur_disponible"] for t in transitions]) if transitions else 0.0

    # --- queue alphabetique --------------------------------------------------
    albums_per_root = d.groupby("Root Genre")["Unique Album"].nunique().to_dict() if has_root else {}
    tail = []
    for v, s, l in reversed(blocks):
        if albums_per_root.get(v, 9) <= 1 or v.lower() == "unknown":
            tail.append((v, s, l))
        else:
            break
    tail.reverse()

    # --- compilations --------------------------------------------------------
    per_album = d.groupby("Album ID").agg(pistes=("Song", "size"), artistes=("Artist", "nunique"),
                                          album=("Album", "first"), genre=("Album Genre", "first"))
    comps = per_album[per_album.artistes >= 3].sort_values("pistes", ascending=False)

    return {
        "fichier": path, "titres": n,
        "cohesion": round(float(np.mean(js)), 4),
        "cohesion_intra": coh_intra,
        "cohesion_inter": coh_inter,
        "ruptures_pct": round(100 * sum(1 for x in js if x == 0) / len(js), 2),
        "sans_genre_pct": round(100 * sum(1 for t in tags if not t) / n, 2),
        "intrus": len(intrus),
        "familles": len(families),
        "familles_eclatees": sum(c - 1 for c in families.values()),
        "macro_obtenu": round(float(got), 4),
        "macro_disponible": round(float(best), 4),
        "macro_ratio": round(float(got / best), 3) if best else None,
        "queue_familles": len(tail),
        "queue_titres": sum(l for _, _, l in tail),
        "compilations_albums": int(len(comps)),
        "compilations_titres": int(comps.pistes.sum()) if len(comps) else 0,
        "_intrus": intrus,
        "_transitions": transitions,
        "_queue": [{"famille": v, "titres": l} for v, _, l in tail],
        "_compilations": comps.head(20).reset_index().to_dict("records"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="CSV produits par le sorter")
    ap.add_argument("--json", help="ecrire le rapport complet dans ce fichier")
    ap.add_argument("--details", action="store_true", help="afficher intrus, queue et compilations")
    args = ap.parse_args()

    reports = []
    header = (f"{'fichier':52}{'titres':>7}{'cohesion':>10}{'inter-alb':>11}{'intrus':>8}"
              f"{'familles':>10}{'eclatees':>10}{'macro':>8}{'queue':>7}{'compil.':>9}")
    print(header)
    print("-" * len(header))
    for path in args.csv:
        try:
            r = audit(path)
        except Exception as exc:                       # noqa: BLE001
            print(f"{path:52}  ERREUR: {exc}", file=sys.stderr)
            continue
        reports.append(r)
        macro = f"{round(100 * r['macro_ratio'])} %" if r["macro_ratio"] is not None else "—"
        inter = f"{r['cohesion_inter']:.3f}" if r.get("cohesion_inter") is not None else "—"
        print(f"{path[-52:]:52}{r['titres']:7}{r['cohesion']:10.3f}{inter:>11}{r['intrus']:8}"
              f"{r['familles']:10}{r['familles_eclatees']:10}{macro:>8}"
              f"{r['queue_titres']:7}{r['compilations_titres']:9}")

    if args.details and reports:
        r = reports[-1]
        print(f"\n--- {r['fichier']} ---")
        print(f"\nQueue alphabetique ({r['queue_familles']} familles, {r['queue_titres']} titres) :")
        print("  " + " · ".join(f"{q['famille']}({q['titres']})" for q in r["_queue"]))
        print("\nFrontieres les plus ratees (ecart obtenu / disponible) :")
        worst = sorted(r["_transitions"], key=lambda t: t["obtenu"] - t["meilleur_disponible"])[:10]
        for t in worst:
            print(f"  {t['de']:26} -> {t['vers']:26} obtenu {t['obtenu']:.3f}  "
                  f"disponible {t['meilleur_disponible']:.3f}")
        if r["_intrus"]:
            print(f"\nIntrus ({r['intrus']}) :")
            for x in r["_intrus"][:15]:
                print(f"  #{x['position']} {x['artiste']} — {x['titre']} [{x['genre']}] "
                      f"entre « {x['precedent']} » et « {x['suivant']} »")
        print("\nCompilations (genre resolu par album, donc faux pour la plupart des pistes) :")
        for c in r["_compilations"][:10]:
            print(f"  {c['pistes']:3} pistes  {c['artistes']:2} artistes  "
                  f"{str(c['album'])[:48]:48} -> {c['genre']}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(reports, fh, ensure_ascii=False, indent=1)
        print(f"\nRapport complet ecrit dans {args.json}")


if __name__ == "__main__":
    main()

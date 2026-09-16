# ---------------------------------------------------------------------------
# ADD THIS AS A NEW CELL IN nb14, AFTER the cell that writes step14_security.csv
#
# Why: the existing cell asks "does the defense return the ORIGINAL clean text?"
# That is the right question for a filter that only deletes zero-width
# characters. bnunicodenormalizer also rewrites ordinary text, so it fails that
# test (82.8%) even where it is perfectly safe.
#
# The question that actually decides security is whether the model receives the
# same input either way: defend(attacked) == defend(clean). If that holds, the
# classifier cannot behave differently, so evasion is 0 by construction -- the
# same argument already used for render-equivalence, and still no GPU.
# ---------------------------------------------------------------------------
rows = []
for a, texts in ATK.items():
    for d, fn in DEFENSES.items():
        dc = [fn(c) for c in hate_txt]                       # defended clean
        da = [fn(x) for x in texts]                          # defended attacked
        rows.append({
            "attack": a,
            "defense": d,
            # model sees identical input for clean and attacked -> evasion 0
            "defended_inputs_identical": round(float(np.mean(
                [p == q for p, q in zip(da, dc)])), 4),
            # kept for comparison with the existing column
            "exact_clean_recovery": round(float(np.mean(
                [p == c for p, c in zip(da, hate_txt)])), 4),
        })
        print(rows[-1])

eq = pd.DataFrame(rows)
eq.to_csv(f"{OUT}/step14_equivalence.csv", index=False)

print("\nIf defended_inputs_identical == 1.0 for a defense, its evasion is 0.0")
print("by construction and the blank cells in the defense table can be filled.")

**Confirmed edge cases / false acceptances**

1. **Malformed 1–3 digit lead groups are silently expanded.** `_EN_GROUPED_INTEGER` allows `[1-9]\d{0,2}`, so `12,345` → “twelve thousand three hundred and forty-five” and `1,234` → “one thousand…”. That's ordinary, but also `99,999` crosses out of range2–5000 into a “thousand” reading. In corpus (2–5000) these should never need expansion, so any `d,ddd` with d beyond 5 is out-of-corpus input being normalized anyway — flag if you want strictness.

2. **`1,000`–`5,000` now equate to spoken “thousand” phrases, but only via the en path.** For `language` values that fall back to `"en"` (any unsupported code is coerced to en before substitution? — no: coercion happens first, so all non-sv become en). That means an `sv`-intended call mis-passed as `de` would get English thousand expansion. Coercion masks caller error rather than warning; correct spoken Swedish `3 619`/`3619` stays numeric and will still mismatch. Not a false acceptance, but a silent-language-fallback divergence worth noting.

3. **Negative sign straddling:** `-3,619` is matched (leading `-?`), good. But `3,619-` (trailing dash, e.g. range) is *not* excluded by the lookahead `(?![\d,:/-]|\.\d)` — the trailing `-` is consumed as boundary only, so `3,619-5,000` expands **both** endpoints to “three thousand… five thousand”, turning a numeric range into two spoken words. Confirm: the regex permits `3,619` when followed by `-` then `5`; the second `5,000` matches via its own `(?<!...)` because preceding char is `-`. If your tests didn't include a hyphenated range, that's a candidate false acceptance: reference `3,619-5,000` may now compare equal to a hypothesis that says only “three thousand six hundred and nineteen five thousand” without range semantics. Likely low-impact in corpus but confirm intent.

4. **Unit alias gap for en:** `"kilogram"`/`"kilogramme"` singular not in alias map, only `"kilogrammes"` and `"kg"`. Spoken “one kilogram” vs ASR `1 kg` → normalized `one kilogram` vs `one kilogram` only if `num2words` yields “kilogram”; but `"kilograms"` alias exists while `"kilogram"` does not — asymmetric. Hypothesis “1 kg” → “one kilograms”? Check: `num2words(1)` = “one”, alias maps `kg`→`kilograms`, giving “one kilograms” vs reference “one kilogram”. **Candidate false rejection**, not acceptance. Verify with a unit test at value 1.

5. **`metres`→`meters`, `metre`→`meter`, but no `"meter"/"meters"` identity needed** — fine.

6. **Terminal integer test lines:** `normalize_text("There are 20, and…")` expects `twenty,` — the regex `(?![\d:/-]|[.,]\d)` allows `20` before `,` + space. Confirmed by test line. No contradiction.

**No confirmed false acceptance found in the listed test lines**; items 3 and 4 are the edge cases to validate. Suggest adding: unit-singular test, hyphenated-range test, and a `99,999` out-of-range guard test.

# NeuralCompanion theme verification

final result: passed

## Scope and visual truth

Match the user's supplied NeuralCompanion style scheme in the existing PySide6
factory. Retain factory page content, navigation, training controls, settings and
runtime behavior. This is a shared-component/theme adaptation, not a recreation
of NeuralCompanion's Persona screen.

Reference: `C:/Users/lainol/AppData/Local/Temp/codex-clipboard-8210cb30-473d-42eb-a025-18977d68509a.png`.
The reference was opened and compared alongside the rendered factory screenshot
in the same comparison input. Its palette was sampled and corroborated against
`H:/PUBLIC MAIN/NeuralCompanion/ui/styles/app.qss` (read only). The two checkbox
SVGs were copied from that application's existing assets, without modifying NC.

## Capture and normalization

- Source: 1769 x 1167 pixels; original display density is unknown.
- Matching-size implementation: `.artifacts/nc-theme-20261104/LFM3-Fine-tuning-1769.png`.
- Target Qt device pixel ratio: 1.0; no image scaling or bitmap editing was used.
- The supplied source includes native title-bar chrome; Qt widget captures omit
  that chrome. Shared component colors, typography, borders and states were
  compared instead of unrelated product content or absolute element positions.
- Additional captures: all nine factory pages at 1250 x 850, plus training/voice
  pages at 1000 x 780.
- Focused evidence: `Voice-Manager-checkbox-checked-1250.png` and
  `Voice-Manager-1000.png` in the same artifact folder, inspected for checkbox
  imagery, form controls, sliders, navigation selection, and text wrapping.

## Fidelity checks

- **Colors:** reference charcoal colors `#11161d`, `#131a23`, `#18202a`, border
  `#283342`, cyan action `#35bfd7`, cyan focus/selection `#66d9ef`, selected fill
  `#233245`, and light text are applied consistently.
- **Typography:** Segoe UI retained, with semibold actions/section headings and
  lighter secondary text. Factory body text remains 13 px for readability; its
  page heading is 22 px. NC's own base stylesheet uses 12 px, so the factory's
  slightly larger body size is an intentional product-specific adjustment.
- **Spacing/layout:** rounded 10–12 px controls/cards and outlined selections
  match the reference language. Existing factory layout and accessible action
  order are retained. Normal/narrow captures show usable sliders and no guided
  page horizontal overflow.
- **Images/assets:** the existing NC active/inactive checkbox SVGs render sharply
  and are packaged with the app. No reference avatars, logos or addon-specific
  imagery were fabricated or inserted into unrelated factory controls.
- **Content:** existing factory text and task flow remain unchanged by this style
  pass. Disabled actions stay visibly disabled; labels are readable on the new
  surfaces, and the main guided controls remain reachable.

## Comparison history

1. Initial capture matched the palette but the 1000 px Voice Manager gained a
   one-pixel horizontal overflow after the new borders/checkbox metrics (P2).
2. Reduced shared card padding from 14 px to 12 px. Re-ran the layout regression
   tests and recaptured both guided pages at 1000 px; horizontal overflow is zero.
3. Recompared the final captures with the reference at full view and focused
   control level. No actionable P0/P1/P2 style differences remain within scope.

## Verification and limits

- Targeted UI/guide/installer checks: 41 passed.
- Final full suite: 180 passed; Ruff and whitespace checks passed.
- Rebuilt installer includes both SVGs; staged-app startup and SVG rendering work
  from a different working directory, without a dependency on the NC installation.
- No actual GPU speech generation, model download, or training was needed for
  this visual-only pass. Screenshots do not establish full accessibility compliance.
- This report covers the UI and installer source prepared for release v0.1.0.

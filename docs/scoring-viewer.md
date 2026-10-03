# Local skin scoring viewer

Open the prepared [32-candidate viewer](../images/results/roadmap-2026-10-02/scoring-viewer/index.html) in Firefox. You can copy just `index.html` anywhere: every model and texture is embedded, with no companion files required. It contains the existing generated development candidates, anonymously reshuffled, with classic geometry and original alpha. No human ratings have been supplied. The local packet is unavailable from a fresh clone.

Browse thumbnails or use Previous/Next. Drag the model with the mouse to rotate it, and use the scroll wheel to zoom. The model background is white. Front/back buttons, automatic rotation and base/overlay toggles remain available; zoom changes display size only. Ratings sit beside the model on laptop-sized screens. On smaller screens, click **Rate this skin** to jump directly to the score inputs. Give five 1–5 ratings, choose accept/edit/discard, and add optional notes. Candidate navigation retains scores and marks fully scored candidates. Reviewer aliases are optional; avoid personal information.

Click **Save rated HTML** to download one self-contained file with every candidate and your current ratings. Reopening that file restores its embedded scores, even with browser storage unavailable. Valid newer browser drafts for the same packet take precedence, so subsequent edits survive reloads. Scores remain editable; save rated HTML again after making changes.

Click **Export scores CSV** for the rubric-compatible sheet, or **Save JSON backup** for a portable draft including candidate hashes, rendering profile, reviewer alias and export time. JSON backups can be restored only into the matching packet; invalid scores and mismatched packets leave existing scores intact. CSV includes every candidate, keeping missing ratings blank, and repeats packet ID, image hash, reviewer alias and the fixed rendering profile in each row. Notes are quoted and spreadsheet formula prefixes are escaped; JSON preserves the original note text.

Browser storage may restore drafts on reload, but storage can be blocked or scoped differently by browser, file location or private mode. Save rated HTML or download a JSON backup before closing or moving the viewer. The browser cannot silently rewrite the original HTML file; explicit download is what makes ratings portable. Use a separate browser profile or clear that packet's site storage when another reviewer needs a blank draft. There is no server or automatic file write, and no data is sent anywhere.

## Create another packet

```bash
scripts/uv.sh sync --locked --group dev
scripts/uv.sh run --locked skin-gan review path/to/skins path/to/new-review \
  --model-type classic --alpha-mode original --seed 1976
```

Open `path/to/new-review/index.html` in Firefox. Supply 1–256 single-frame 64×64 PNGs, each below 4 MiB. The output must be a new directory. PNG bytes are copied unchanged; source files remain intact. `--model-type slim` supports narrow arms; `--alpha-mode opaque-base` renders base alpha opaque without changing exported PNGs. Fix these settings before scoring and keep them identical across comparisons.

`review.json` records packet identity and original-byte hashes. `candidate-map.json` is a **researcher-only source mapping**: keep it closed until scoring and thresholds are fixed, and do not distribute it to blinded reviewers. The viewer and score exports contain anonymous IDs and hashes rather than source filenames or training labels. Preserve the map privately to join scores to experiments later. Anonymous naming cannot undo prior visual exposure; these previously inspected outputs remain development evidence.

The scoring interface implements the [draft rubric](quality-rubric.md); it does not approve quality thresholds, invent reviewer judgments, establish skin originality, or certify actual Minecraft acceptance. Batch variety remains a separate batch-level assessment after per-skin scoring and unblinding, as specified in that rubric.

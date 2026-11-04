# Tools for measuring image-to-text linking

These scripts let you check, with numbers, whether the system links each picture to the right
part of the text. Nothing here changes the chatbot.

## 1. Create a template for a PDF

```bash
python tools/make_gold_template.py path/to/paper.pdf
```

This writes `gold/<name>.gold.json` and the picture crops in `gold/<name>_review/crops/`.

## 2. Label it (about 10 minutes per document)

Open the JSON. For each entry in `figures`:

- Look at the crop `gold/<name>_review/crops/<figure_id>.png`.
- Find the section(s) in the `sections` list whose **text discusses that picture** and put their ids
  in `relevant_sections`, for example `["s015"]`.
- If nothing in the document discusses the picture, set `"no_related_text": true` instead.
- Leave a figure untouched to skip it.

`located_in_section` shows where the picture sits on the page. It is often, but not always, the
right answer - a picture can be discussed in a different section.

Aim for 20-30 labelled figures across 3-5 different documents (a science chapter, a paper,
a story, ...).

## 3. Measure

```bash
python tools/evaluate_links.py gold
python tools/evaluate_links.py gold --linker signals --min-score 0.3 -v
```

Output columns: `top1` / `top3` = the correct section is the best / among the 3 best links;
`MRR` = average of 1/rank of the correct section; `wrong` = confident but wrong;
`not-sure` = the linker declined (score below `--min-score`); `no-text-correct` = for pictures
with no related text, how often it correctly declined.

`baseline` links each picture to the nearest text. Any smarter method has to beat it.

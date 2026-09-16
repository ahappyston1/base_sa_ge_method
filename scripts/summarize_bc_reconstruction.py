"""Read-only, standard-library summary of BC reconstruction diagnostics."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def histogram_summary(histogram):
    """Binned AUROC: larger error predicts WRONG pseudo-labels; ties get 1/2.

    Scores within one 0.1-wide bin are treated as tied. This is an approximate
    AUROC over visits, not independent samples and not a calibrated probability.
    """
    good = sum(v[0] for v in histogram.values())
    bad = sum(v[1] for v in histogram.values())
    lower_good = favorable = 0.
    for index in sorted(histogram):
        g, b = histogram[index]
        favorable += b*(lower_good+.5*g)
        lower_good += g
    def mean(which, count):
        return sum((index+.5)*.1*values[which] for index, values in histogram.items())/count if count else None
    return dict(correct_visits=good, wrong_visits=bad,
                binned_error_auroc=favorable/(good*bad) if good and bad else None,
                correct_error_midpoint_mean=mean(0, good), wrong_error_midpoint_mean=mean(1, bad))


def summarize(run_dir, start, end):
    histograms = defaultdict(lambda: defaultdict(lambda: [0., 0.]))
    with (run_dir/'reconstruction_audit.csv').open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if not start <= int(row['round']) <= end:
                continue
            keys = [(row['score'], 'all', 'all', 'all'),
                    (row['score'], row['predicted_class'], row['bucket'], row['confidence_band'])]
            for key in keys:
                counts = histograms[key][int(row['error_bin'])]
                counts[0] += float(row['correct'])
                counts[1] += float(row['wrong'])
    references = defaultdict(lambda: dict(query_count=0., query_correct=0., own_distance_sum=0., margin_sum=0.))
    with (run_dir/'reconstruction_reference.csv').open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if start <= int(row['round']) <= end:
                for name in references[row['cls']]:
                    references[row['cls']][name] += float(row[name])
    geometry = []
    for cls, values in sorted(references.items()):
        n = values['query_count']
        geometry.append(dict(cls=int(cls), **values,
                             query_accuracy=values['query_correct']/n if n else None,
                             mean_own_distance=values['own_distance_sum']/n if n else None,
                             mean_margin=values['margin_sum']/n if n else None))
    return dict(source=str(run_dir.resolve()), start=start, end=end,
                interpretation='Counts are repeated visits. AUROC is binned, not calibrated; queries are held out only from centroid fitting.',
                reconstruction=[dict(score=k[0], predicted_class=k[1], bucket=k[2], confidence_band=k[3],
                                     **histogram_summary(hist)) for k, hist in sorted(histograms.items())],
                reference=geometry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--start', type=int, default=91)
    parser.add_argument('--end', type=int, default=300)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.start <= args.end:
        parser.error('Require 1 <= start <= end')
    result = summarize(args.run_dir, args.start, args.end)
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(output+'\n', encoding='utf-8')
    else:
        print(output)


if __name__ == '__main__':
    main()

import collections
import datetime
import json
import pathlib
import sys

from server import sface_results


result = sface_results()
summary = {}
for name, rows in (('enrollment', result['rows'][:1]),
                   ('same', result['rows'][1:30]),
                   ('different', result['rows'][30:])):
    summary[name] = dict(collections.Counter(row['status'] for row in rows))
result['summary'] = summary
timestamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H-%M-%S-%fZ')
output = pathlib.Path(__file__).resolve().parent / f'results-sface-{timestamp}.json'
with output.open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(f'Saved {output}')
print(json.dumps(summary, indent=2))
if summary['enrollment'].get('enrolled') != 1 or any(
    row['status'] in ('missing', 'processing error', 'enrollment unavailable')
    for row in result['rows']
):
    sys.exit(1)

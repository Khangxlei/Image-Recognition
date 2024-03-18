import linecache
import tracemalloc
import sys
import subprocess
import cProfile, pstats, io
from pstats import SortKey

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    # Get statistics based on specified key type
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    
    # Iterate over the top memory-consuming lines
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    # Print statistics for other lines if present
    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

# Start tracing memory allocations
tracemalloc.start()

#Start profiling
pr = cProfile.Profile()
pr.enable()

# Execute test_matmul.py script using subprocess

    
python_path = sys.executable  # Get the path to the current Python interpreter
subprocess.run([python_path, "../tests/data_upload_test.py"], check=True)
subprocess.run([python_path, "../tests/model_upload_test.py"], check=True)
subprocess.run([python_path, "../tests/train_test.py"], check=True)
subprocess.run([python_path, "../tests/inference_test.py"], check=True)

pr.disable()
s = io.StringIO()

# Create a profiler statistics object and print the profiling results
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

# Take a snapshot of memory allocations after the execution
snapshot = tracemalloc.take_snapshot()
display_top(snapshot)

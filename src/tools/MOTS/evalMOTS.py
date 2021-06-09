# TODO add proper citations
# author = {Paul Voigtlaender and Michael Krause and Aljo\u{s}a O\u{s}ep and Jonathon Luiten and Berin Balachandar Gnana Sekar and Andreas Geiger and Bastian Leibe}
# Modified by Gaspar Faure

import sys, os
import argparse
sys.path.append(os.path.abspath(os.getcwd()))
import math
from collections import defaultdict
from MOTS_metrics import MOTSMetrics
from Evaluator import Evaluator, run_metrics

import multiprocessing as mp

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--benchmark_name', type=str, default='MOTS')
    parser.add_argument('--gt_dir', type=str, default='data/MOTS', help='Directory containing ground truth files.')
    parser.add_argument('--res_dir', type=str, default='res/MOTSres', help='Directory containing result files.')
    parser.add_argument('--seqmaps_dir', type=str, default='seqmaps', help='Directory containing txt file containing sequences to eval')
    parser.add_argument('--eval_mode', type=str, default='train', help='Can be [train | test | all]')

    return parser.parse_args()

class MOTS_evaluator(Evaluator):
	def __init__(self):
		self.type = "MOTS"
	def eval(self):

		arguments = []
		for seq, res, gt in zip(self.sequences, self.tsfiles, self.gtfiles):
			arguments.append({"metricObject": MOTSMetrics(seq),
			"args" : {"gtDataDir": os.path.join( self.datadir,seq),
			"sequence": str(seq) ,
			"pred_file":res,
			"gt_file": gt,
			"benchmark_name": self.benchmark_name}})

		if self.MULTIPROCESSING:
			p = mp.Pool(self.NR_CORES)
			processes = [p.apply_async(run_metrics, kwds=inp) for inp in arguments]
			self.results = [p.get() for p in processes]
			p.close()
			p.join()

		else:
			results = [run_metrics(**inp) for inp in arguments]


		# Sum up results for all sequences
		self.Overall_Results = MOTSMetrics("OVERALL")


if __name__ == "__main__":
	args = parse_args()
	eval = MOTS_evaluator()
	benchmark_name = args.benchmark_name
	gt_dir = args.gt_dir
	res_dir = args.res_dir
	eval_mode = args.eval_mode
	seqmaps_dir = args.seqmaps_dir
	eval.run(
	         benchmark_name = benchmark_name,
	         gt_dir = gt_dir,
	         res_dir = res_dir,
	         eval_mode = eval_mode,
			 seqmaps_dir = seqmaps_dir)

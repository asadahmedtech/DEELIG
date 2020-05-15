#!/usr/bin/python

#to run PSI-BLAST for several queries on a given database
#/home/user/ptp/scripts/psi_blast.py

import os
import argparse
import glob

parser = argparse.ArgumentParser(description='PSIBLAST for several queries against a database')
parser.add_argument('-query', dest='query_dir',required=True , help='path to psiblast query')
parser.add_argument('-out', dest='out_dir',required=True , help='path to psiblast output')
parser.add_argument('-db', dest='database_dir', required=True , help='path to database')
parser.add_argument('-evalue', dest='evalue', type=float,default=0.00001, help='E-value')
parser.add_argument('-inclusion_ethresh', dest='inclusion_ethresh', type=float, default=0.1, help='inclusion threshold')
parser.add_argument('-num_iterations', dest='num_iterations', type=int, default=20, help='number of iterations')
parser.add_argument('-num_threads', dest='num_threads', type=int, default=10, help='number of threads')
args = parser.parse_args()


in_dir=args.query_dir
db_file=args.database_dir
evalue=args.evalue
ethresh=args.inclusion_ethresh
iterations=args.num_iterations
threads=args.num_threads

for query_file_path in glob.glob(in_dir+"/*.fasta") :

        base_name = os.path.basename(query_file_path)
        out_path = "%s/%s.out"%(args.out_dir,os.path.splitext(base_name)[0])

        cmd_str="/home/mini/teernab/ptp/mammalian-genome-search/ncbi-blast-2.3.0+/bin/psiblast -query %s -db %s -out %s -evalue %s -inclusion_ethresh %s -num_iterations %s -outfmt '6 qseqid sseqid length qstart qend sstart send bitscore evalue qcovs pident' -num_threads %s"\
                                %(query_file_path, db_file, out_path, evalue, ethresh, iterations, threads)

        print cmd_str
        os.system(cmd_str)


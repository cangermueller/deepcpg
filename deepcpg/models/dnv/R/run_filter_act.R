#!/usr/bin/env Rscript

library(argparse)
library(rmarkdown)
library(tools)

p <- ArgumentParser(description='Evaluate filter activations')
p$add_argument(
  'db_file',
  help='Filter evaluation file')
p$add_argument(
  '-o',
  '--out_file',
  default='filter_act.html',
  help='Output file')
p$add_argument(
  '--verbose',
  action='store_true',
  help='More detailed log messages')

args <- commandArgs(TRUE)
opts <- p$parse_args(args)
if (!is.null(opts$verbose)) {
  print(opts)
}

opts$db_file <- file_path_as_absolute(opts$db_file)

rmd <- file.path(Sys.getenv('Pv'), 'R', 'filter_act.Rmd')
out_format <- paste(file_ext(opts$out_file), 'document', sep='_')

rmarkdown::render(rmd, output_file=opts$out_file,
  output_format=out_format,
  output_dir=dirname(opts$out_file))

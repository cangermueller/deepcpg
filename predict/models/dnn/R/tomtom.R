#!/usr/bin/env Rscript

library(argparse)
library(dplyr)
library(stringr)

lib <- file.path(Sys.getenv('Pd'), 'R')
source(file.path(lib, 'utils.R'))
source(file.path(lib, 'io.R'))

p <- ArgumentParser(description='Formats tomtom csv file')
p$add_argument(
  'in_file',
  help='Input file')
p$add_argument(
  '-o',
  '--out_file',
  help='Output file')
p$add_argument(
  '--verbose',
  action='store_true',
  help='More detailed log messages',
  default=F)

args <- commandArgs(TRUE)
if (length(args) > 0) {
  opts <- p$parse_args(args)
} else {
  opts <- list()
}
if (opts$verbose) {
  print(opts)
}

dat <- list()
dat$tomtom <- read_tomtom(opts$in_file, all=T)
out_file <- opts$out_file
if (is.null(out_file)) {
  out_file <- './tomtom_nice.csv'
}
write.table(dat$tomtom, out_file, sep='\t')



#!/usr/bin/env Rscript


library(argparse)
library(ggplot2)
library(dplyr)
library(tidyr)
library(rhdf5)
library(grid)
library(seqLogo)

lib <- file.path(Sys.getenv('Pd'), 'R')
source(file.path(lib, 'filter.R'))


p <- ArgumentParser(description='Visualize filters')
p$add_argument(
  'filt_file',
  help='Input file')
p$add_argument(
  '--weights',
  default='/filter/weights',
  help='HDF path to filter weights')
p$add_argument(
  '-o', '--out_motifs',
  default='./motifs.pdf',
  help='Name of output motifs file')
p$add_argument(
  '--out_heat',
  help='Name of output heatmap file')
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
  opts$filt_file <- './filters.h5'
  opts$weights <- '/filter/weights'
  opts$out_motifs <- './filter_motifs.pdf'
  opts$out_heat <- './filter_heat.pdf'
  opts$verbose <- 0
}

log <- function(x) {
  cat(x, fill=T)
}

if (opts$verbose == 1) {
  print(opts)
}

dat <- list()
log('Read filters')
dat$filt <- read_filt(opts$filt_file, group=opts$weights)
dat$nb_filt <- length(levels(dat$filt$filt))
dat$filt_len <- max(as.numeric(dat$filt$pos))


if (!is.null(opts$out_heat)) {
  log('Plot heat map')
  p <- plot_filt_heat(dat$filt, 'act_ms')
  ggsave(p, file=opts$out_heat,
    width=dat$filt_len * 0.8 * 3,
    height=ceiling(dat$nb_filt / 3) * 2
    )
}

if (!is.null(opts$out_motifs)) {
  log('Plot sequence motifs')
  pdf(opts$out_motifs,
    width=dat$filt_len * 1.0,
    height=5
    )
  for (i in 1:dat$nb_filt) {
    d <- filt_pwm(dat$filt, i - 1, 'act_p2')
    seqLogo(d)
  }
  dev.off()
}

log('Done!')

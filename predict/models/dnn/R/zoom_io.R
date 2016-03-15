h5ls <- function(path, group='/') {
  if (group[1] != '/') {
    group <- paste0('/', group)
  }
  h <- sprintf('h5ls %s%s', path, group)
  h <- system(h, intern=T)
  h <- sapply(strsplit(h, '\\s+'), function(x) x[1])
  return (h)
}

read_list <- function(path, group) {
  h <- h5ls(path=path, group)
  d <- lapply(h, function(x) h5read(path, file.path(group, x)))
  names(d) <- h
  return (d)
}

read_h5 <- function(path, targets_regex=NULL, chromos_regex=NULL) {
  targets <- h5ls(path)
  if (!is.null(targets_regex)) {
    targets <- grep(targets_regex, targets, value=T)
  }
  ds <- list()
  for (target in targets) {
    chromos <- h5ls(path, group=target)
    if (!is.null(chromos_regex)) {
      chromos <- grep(chromos_regex, chromos, value=T)
    }
    for (chromo in chromos) {
      p <- file.path(target, chromo)
      h <- h5ls(path, group=p)
      d <- lapply(h, function(x) h5read(path, file.path(p, x)))
      names(d) <- h
      d <- as.data.frame(d)
      d$target <- target
      d$chromo <- chromo
      ds[[length(ds) + 1]] <- d
    }
  }
  d <- do.call(rbind.data.frame, ds) %>%
    move_cols(c('chromo', 'pos', 'target')) %>%
    char_to_factor %>% tbl_df
  if ('y' %in% names(d)) {
    h <- d$y == -1
    d$y[h] <- NA
  }
  return (d)
}

parse_seqmut <- function(x) {
  return (sub('.*_([^-0123456789]+).*', '\\1', x))
}

parse_seqmut_wlen <- function(x) {
  return (sub('.*?([-0123456789]+)$', '\\1', x))
}

within_01 <- function(x, eps=1e-3) {
  return (pmin(1 - eps, pmax(eps, x)))
}

lor <- function(x, y) {
  x <- within_01(x)
  y <- within_01(y)
  h <- log2(x / (1 - x)) - log2(y / (1 - y))
  return (h)
}

format_seqmut <- function(d) {
  h <- levels(d$zx)
  d <- d %>% mutate(
      seqmut=factor(zx, levels=h, labels=parse_seqmut(h)),
      wlen=factor(zx, levels=h, labels=parse_seqmut_wlen(h)),
      del = value - z,
      lor = lor(value, z)
    )
  return (d)
}

parse_filt <- function(x) {
  return (as.numeric(sub('.*f(\\d+)', '\\1', x)))
}

format_filt_imp <- function(d) {
  h <- levels(d$zx)
  d <- d %>% mutate(
      filt=factor(zx, levels=h, labels=parse_filt(h)),
      del = z - value,
      lor = lor(z, value)
    )
  return (d)
}

parse_filt_act <- function(x) {
  x <- as.numeric(sub('V', '', x)) - 1
  stopifnot(min(x) == 0)
  return (x)
}

read_bed <- function(path, chromo=NULL) {
  d <- read.table(path)
  names(d) <- c('chromo', 'start', 'end')
  d <- d %>% tbl_df
  return (d)
}

bed_to_atrack <- function(d) {
  a <- AnnotationTrack(
    chromo=d$chromo,
    start=d$start,
    end=d$end)
  return (a)
}

filter_bed <- function(d, region) {
  d <- d %>% filter(
    chromo==as.character(region$chromo), start >= region$start, end <= region$end
    )
  return (d)
}

beds_to_atracks <- function(d, region) {
  a <- list()
  for (n in names(d)) {
    dn <- d[[n]] %>% filter_bed(region)
    if (nrow(dn) == 0) {
      next
    }
    an <- bed_to_atrack(dn)
    an@name <- n
    a[[n]] <- an
  }
  return (a)
}

read_filt_act <- function(path, chromo=NULL) {
  d <- list()
  d$chromo <- h5read(path, '/chromo')
  d$pos <- h5read(path, '/pos')
  if (!is.null(chromo)) {
    idx <- which(d$chromo == chromo)
  }
  for (n in c('chromo', 'pos')) {
    d[[n]] <- d[[n]][idx]
  }
  d$act <- t(h5read(path, '/act', index=list(NULL, idx)))
  return (d)
}

filter_filt_act <- function(d, chromo=NULL, start=NULL, end=NULL,
  nb_filt=NULL, region=NULL) {
  idx <- rep(T, nrow(d$act))
  if (!is.null(region)) {
    chromo <- region$chromo
    start <- region$start
    end <- region$end
  }
  if (!is.null(chromo)) {
    chromo <- sub('^chr', '', tolower(chromo))
    idx <- idx & (d$chromo == chromo)
  }
  if (!is.null(start)) {
    idx <- idx & (d$pos >= start)
  }
  if (!is.null(end)) {
    idx <- idx & (d$pos <= end)
  }
  act <- d$act[idx,]
  colnames(act) <- sapply(1:ncol(act), function(x) sprintf('%d', x - 1))
  if (!is.null(nb_filt)) {
    nb_filt <- min(nb_filt, ncol(act))
    s <- rev(order(colMeans(act)))[1:nb_filt]
    act <- act[, s]
  }
  act <- act %>% as.data.frame
  act$pos <- d$pos[idx]
  act$chromo <- d$chromo[idx]
  act <- act %>% arrange(chromo, pos) %>%
    move_cols(c('chromo', 'pos')) %>%
    gather(filt, value, -c(chromo, pos)) %>%
    tbl_df
  return (act)
}

select_filt_fun <- function(act, fun=mean, nb_filt=10) {
  h <- act %>% group_by(filt) %>% summarise_each(funs(fun), value) %>%
    ungroup %>% arrange(desc(value)) %>% select(filt) %>% unlist %>% as.vector
  if (!is.null(nb_filt)) {
    h <- h[1:nb_filt]
  }
  act <- act %>% filter(filt %in% h) %>%
    mutate(filt=factor(filt, levels=h))
  return (act)
}

select_filt_cor <- function(act, y, what='z', nb_filt=10) {
  y$y <- y[[what]]
  y <- y %>% select(chromo, pos, y)
  h <- act %>% inner_join(y, by=c('chromo', 'pos'))
  stopifnot(nrow(h) == nrow(act))
  act <- h
  h <- act %>% group_by(filt) %>%
    summarise(value=abs(cor(value, y, method='spearman'))) %>% ungroup %>%
    arrange(desc(value)) %>% select(filt) %>% unlist %>% as.vector
  if (!is.null(nb_filt)) {
    h <- h[1:nb_filt]
  }
  act <- act %>% filter(filt %in% h) %>%
    mutate(filt=factor(filt, levels=h))
  return (act)
}

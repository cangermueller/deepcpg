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

read_h5 <- function(path) {
  targets <- h5ls(path)
  targets <- grep('^ESC', targets, value=T)
  ds <- list()
  for (target in targets) {
    chromos <- h5ls(path, group=target)
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
    gather(zx, value, starts_with('z_')) %>%
    move_cols(c('chromo', 'pos', 'target')) %>%
    char_to_factor %>%
    mutate(
      cell_type=parse_cell_type(target)
      ) %>% tbl_df
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

read_filt_act <- function(path, group='/act/s_x', chromo=NULL, wlen=101) {
  if (is.null(chromo)) {
    chromos <- h5ls(path, group)
  } else {
    chromos <- c(chromo)
  }
  ds <- list()
  del <- as.integer(wlen / 2)
  for (chromo in chromos) {
    h <- file.path(group, chromo)
    h <- read_list(path, h)
    y <- h$y
    m <- as.integer(dim(y)[2] / 2)
    y <- y[,(m - del):(m + del),]
    y <- apply(y, c(1, 3), mean)
    d <- as.data.frame(t(y))
    d$pos <- h$pos
    d <- d %>% gather(filt, value, -pos)
    d$chromo <- chromo
    ds[[length(ds) + 1]] <- d
  }
  d <- do.call(rbind.data.frame, ds) %>% tbl_df
  h <- levels(d$filt)
  d <- d %>% mutate(filt=factor(filt, levels=h, labels=parse_filt_act(h)))
  d <- d %>% move_cols(c('chromo', 'pos', 'filt'))
  return (d)
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
    chromo==region$chromo, start >= region$start, end <= region$end
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

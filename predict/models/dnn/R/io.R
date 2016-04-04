query_db <- function(db_file, table='global', cond=NULL) {
  con <- src_sqlite(db_file)
  h <- sprintf('SELECT * FROM %s', table)
  if (!is.null(cond)) {
    h <- sprintf('%s WHERE %s', h, cond)
  }
  d <- tbl(con, sql(h))
  d <- d %>% collect %>% select(-path, -id)
  if ('filt' %in% names(d)) {
    d$filt <- factor(d$filt)
  }
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df
  return (d)
}

query_dbs <- function(db_path) {
  g <- query_db(db_path, 'global') %>% mutate(anno='global')
  a <- query_db(db_path, 'annos')
  g <- g[, names(a)]
  a <- rbind.data.frame(g, a) %>%
    mutate(anno=factor(anno)) %>%
    droplevels
  return (a)
}

join_prefix <- function(d, by, prefix, sep='_') {
  n <- names(d)
  h <- n %in% by
  n[!h] <- paste(prefix, n[!h], sep=sep)
  names(d) <- n
  return (d)
}

join_all <- function(dat, by, fun=left_join, prefix=F) {
  d <- dat[[1]]
  if (prefix) {
    d <- d %>% join_prefix(by, names(dat)[1])
  }
  for (i in 2:length(dat)) {
    h <- dat[[i]]
    if (prefix) {
      h <- h %>% join_prefix(by, names(dat)[i])
    }
    d <- d %>% fun(h, by=by)
  }
  return (d)
}

join_datasets <- function(dat) {
  d <- list()
  d$cpg <- dat$cpg %>% group_by(anno, filt) %>%
    summarise(
      cpg_act_mean=mean(act_mean),
      cpg_act_std=mean(act_std),
      cpg_rs=mean(rs)
      ) %>% ungroup

  d$var <- dat$var %>% group_by(anno, filt) %>%
    summarise(
      var_act_mean=mean(act_mean),
      var_act_std=mean(act_std)
      ) %>% ungroup

  d$var2 <- dat$var %>% filter(metric %in% c('var', 'mean')) %>%
    select(anno, filt, metric, rs) %>%
    spread(metric, rs) %>% group_by(anno, filt) %>%
    summarise(var_rs_var=mean(var), var_rs_mean=mean(mean)) %>%
    ungroup %>% mutate(var_rs_del=abs(var_rs_var)-abs(var_rs_mean))

  d$cons <- dat$cons %>% select(anno, filt, cons_rs=rs)

  d <- join_all(d, c('anno', 'filt'))

  if ('tomtom' %in% names(dat)) {
    h <- dat$tomtom %>% join_prefix(by='filt', 'tom')
    d <- d %>% left_join(h, by='filt')
    d$label <- label_filt(d$filt, d$tom_name)
  }
   if ('filt' %in% names(dat)) {
     h <- dat$filt %>% select(filt, motif, ic)
     d <- d %>% left_join(h, by='filt')
   }

  return (d)
}

tomtom_target_name <- function(target.name, target.id) {
  target.name <- as.character(target.name)
  target.id <- as.character(target.id)
  s <- gsub(' ', '_', target.name)
  s <- sapply(str_split(s, '_'), function(x) x[1])
  h <- str_length(s) == 0
  s[h] <- target.id[h]
  s <- gsub('[()]', '', s)
  return (s)
}

read_tomtom <- function(path, all=F) {
  d <- read.table(path, sep='\t', head=T) %>% tbl_df %>%
    mutate(filt=factor(sub('filter', '', Query.ID))) %>%
    group_by(filt) %>% arrange(q.value) %>% ungroup %>%
    mutate(name=factor(tomtom_target_name(Target.name, Target.ID))) %>%
    move_cols(c('filt', 'Target.ID', 'Target.name', 'name',
        'p.value', 'E.value', 'q.value', 'URL'))
  if (!all) {
    h <- d %>% group_by(filt) %>% arrange(q.value) %>% slice(1) %>% ungroup
      # select(filt, Target.name, name, Target.ID, p.value, E.value, q.value, URL)
    d <- h %>% inner_join(group_by(d, filt) %>% summarise(nb_hits=n()))
  }
  return (d)
}

select_tomtom <- function(dat) {
  d <- dat %>% group_by(filt) %>% arrange(q.value) %>% slice(1) %>%
    ungroup
  d <- d %>% select(filt, name=name, q.value=q.value, url=URL)
  h <- dat %>% group_by(filt) %>% arrange(q.value) %>%
    summarise(nb_hit=n(), hits=paste(name, collapse=' '))
  d <- d %>% left_join(h, by='filt')
  return (d)
}

label_tomtom <- function(d, tomtom, by_='filt') {
  d <- d %>% left_join(select(tomtom, filt, label=name), by=by_) %>%
    mutate(
      label=sprintf('%d: %s', as.numeric(as.vector(filt)), label),
      label=sub(': NA', '', label),
      label=factor(label)
      )
  return (d)
}

label_filt <- function(filt, name) {
  d <- c()
  for (i in 1:length(filt)) {
    h <- sprintf('%02d: %s', as.numeric(filt[i]), name[i])
    d <- c(d, h)
  }
  d <- sub(': NA', '', d)
  return (d)
}

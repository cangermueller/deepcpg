query_db <- function(db_file, table='global', cond=NULL) {
  con <- src_sqlite(db_file)
  h <- sprintf('SELECT * FROM %s', table)
  if (!is.null(cond)) {
    h <- sprintf('%s WHERE %s', h, cond)
  }
  d <- tbl(con, sql(h))
  d <- d %>% collect %>% select(-path, -id)
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df
  return (d)
}

read_tomtom <- function(path, all=F) {
  h <- read.table(path, sep='\t', head=T) %>% tbl_df %>%
    mutate(filt=factor(sub('filter', '', Query.ID))) %>%
    group_by(filt) %>% arrange(q.value) %>% ungroup
  if (all) {
    return (h)
  }
  d <- h %>% group_by(filt) %>% arrange(q.value) %>% slice(1) %>% ungroup %>%
    select(filt, Target.name, Target.ID, p.value, E.value, q.value, URL)
  d <- d %>% inner_join(group_by(h, filt) %>% summarise(nb_hits=n()))
  return (d)
}

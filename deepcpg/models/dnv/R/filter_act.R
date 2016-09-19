query_db <- function(db_file, table='global', cond=NULL, value='rs') {
  con <- src_sqlite(db_file)
  h <- sprintf('SELECT * FROM %s', table)
  if (!is.null(cond)) {
    h <- sprintf('%s WHERE %s', h, cond)
  }
  d <- tbl(con, sql(h))
  d <- d %>% collect %>% select(-path, -id)
  d <- d %>% format_target
  d <- d %>% filter(!is.na(rs), !is.na(rp))
  d$value_del <- d[[value]]
  d$value_abs <- abs(d$value_del)
  d <- d %>% char_to_factor %>% droplevels %>% tbl_df
  return (d)
}

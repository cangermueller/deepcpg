#!/usr/bin/env bash

sqlite3 -column $1 'SELECT model, AVG(auc) as auc FROM global GROUP BY model ORDER BY auc DESC'


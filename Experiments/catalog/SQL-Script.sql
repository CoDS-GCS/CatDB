select concat(tab.table_name,':','PK: ',sta.column_name,' FK: ')

from information_schema.tables as tab
inner join information_schema.statistics as sta
        on sta.table_schema = tab.table_schema
        and sta.table_name = tab.table_name
        and sta.index_name = 'PRIMARY'
where tab.table_schema = 'Yelp'
    and tab.table_type = 'BASE TABLE'
order by tab.table_name;


SELECT
    concat(TABLE_NAME,': (', COLUMN_NAME,') references ', REFERENCED_TABLE_NAME, ' (',REFERENCED_COLUMN_NAME,'),')
FROM
  INFORMATION_SCHEMA.KEY_COLUMN_USAGE
WHERE
  REFERENCED_TABLE_SCHEMA = 'Yelp' 



select TABLE_NAME, concat('columns: ', group_concat(column_name order by ordinal_position))
from information_schema.columns
where table_schema = 'Yelp'
group by  TABLE_NAME; -- and table_name = 'table_name'
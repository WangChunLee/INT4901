SELECT CONCAT('"',uid,'","',book_id,'","',MAX(score),'"')
from
(
    select v.uid, v.book_id, v.view_count + (case when fav.id is null then 0 else 3 end) as score
    from (
        select uid, book_id, count(*) as view_count
        from view_record
        group by uid, book_id
    ) as v
    left join fav
    on v.book_id = fav.id

    UNION ALL

    select fav.uid, fav.id as book_id, (case when v.view_count is null then 0 else v.view_count+3 end) as score
    from (
        select uid, book_id, count(*) as view_count
        from view_record
        group by uid, book_id
    ) as v
    right join fav
    on v.book_id = fav.id
    group by fav.uid, fav.id
) b
WHERE score>0
GROUP BY uid, book_id
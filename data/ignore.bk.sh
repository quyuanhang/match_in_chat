# !/usr/bin/sh
fp=$1
start_time=`date -d '-30 day' +%Y-%m-%d`
end_time=`date -d '-10 day' +%Y-%m-%d`
end_time_plus=`date -d '-7 day' +%Y-%m-%d`
n_positive=100000
n_negative=100000

hive -e "
add jar /data1/nlp/oceanus-online/udf/lib/oceanus-common-1.0.8-SNAPSHOT.jar;
add jar /data1/nlp/oceanus-online/udf/lib/oceanus-etl-hive-udf-1.0-SNAPSHOT.jar;
add jar /data1/nlp/oceanus-online/udf/lib/oceanus-nltk-common-1.0.8-SNAPSHOT.jar;
add jar /data1/nlp/oceanus-online/udf/lib/oceanus-nltk-segment-1.0.8-SNAPSHOT.jar;
create temporary function tokenizer as 'com.techwolf.oceanus.etl.hive.udf.TokenizerArrayUDF';

select '======================= selecting geek profile =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_geek as
  select
    geek_id user_id, desc
  from
    dw_bosszp.zp_geek;

create temporary table nlp.qyh_project as
  select
    user_id, concat_ws(' ',collect_set(string(description))) prj
  from
    dw_bosszp.zp_project
  group by
    user_id;

create temporary table nlp.qyh_edu as
  select
    user_id, concat_ws(' ',collect_set(string(edu_desc))) edu
  from
    dw_bosszp.zp_edu
  group by
    user_id;

create temporary table nlp.qyh_work as
  select
    user_id, concat_ws(' ', collect_set(memo)) workdesc
  from
    dw_bosszp.zp_work
  group by
    user_id;

create temporary table nlp.qyh_resume as
  select
    distinct(a.user_id) geek_id, a.desc, b.prj, c.edu, d.workdesc
  from
    nlp.qyh_geek a
  left outer join
    nlp.qyh_project b
  on
    a.user_id = b.user_id
  left outer join
    nlp.qyh_edu c
  on
    a.user_id = c.user_id
  left outer join
    nlp.qyh_work d
  on
    a.user_id = d.user_id;

select '======================= selecting job profile =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_job as
select
  distinct(id) job_id, position, job_desc
from
  dw_bosszp.zp_job;

select '======================= geek start =============================' from dw_bosszp.zp_geek limit 1;
select '======================= selecting all samples =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_geek_add as
  select 
    uid geek_id, actionp2 job_id, actionp boss_id, action
  from
    dw_bosszp.bg_action
  where
    ds > '$start_time' and ds <= 'end_time'
  and
    action = 'detail-geek-addfriend'
  and
    bg = 0
  group by
    uid, actionp2, actionp, action;

select '======================= selecting positive samples =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_boss_chat as
  select 
    actionp geek_id, uid boss_id, action
  from
    dw_bosszp.bg_action
  where
    ds > '$start_time' and ds <= '$end_time_plus'
  and
    action = 'chat'
  and
    bg = 1
  group by 
    actionp, uid, action;

select '======================= joining chat with add friend =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_geek_add_boss_chat as
  select 
    geek_id, job_id, action 
  from (
    select
      a.geek_id, a.job_id, 1 action
    from
      nlp.qyh_sample_geek_add a
    join
      nlp.qyh_sample_boss_chat b
    on
      a.geek_id = b.geek_id and a.boss_id = b.boss_id
  ) a
  group by
    a.geek_id, a.job_id, action;

select '======================= selecting add without chat =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_geek_add_no_chat as
  select 
    geek_id, job_id, boss_id
  from (
    select
      a.geek_id, a.job_id, a.boss_id, b.action
    from
      nlp.qyh_sample_geek_add a
    left join
      nlp.qyh_sample_boss_chat b
    on
      a.geek_id = b.geek_id and a.boss_id = b.boss_id
  ) c
  where
    action is null
  group by
    geek_id, job_id, boss_id;

select '======================= selecting active =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_active_boss as
  select 
    uid boss_id
  from
    dw_bosszp.bg_action
  where
    ds > '$start_time' and ds <= '$end_time_plus'
  and
    action = 'list-geek'
  and
    bg = 1
  group by
    uid;

select '======================= selecting active and no chat =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_geek_add_no_chat_active as
  select
    a.geek_id, a.job_id, 0 action
  from
    nlp.qyh_sample_geek_add_no_chat a
  join
    nlp.qyh_sample_active_boss b
  on
    a.boss_id = b.boss_id;

select '======================= geek end =============================' from dw_bosszp.zp_geek limit 1;

select '======================= job start =============================' from dw_bosszp.zp_geek limit 1;
select '======================= selecting all samples =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_boss_add as
  select 
    actionp geek_id, actionp2 job_id, uid boss_id, action
  from
    dw_bosszp.bg_action
  where
    ds > '$start_time' and ds <= 'end_time'
  and
    action = 'detail-geek-addfriend'
  and
    bg = 1
  group by
    actionp, actionp2, uid, action;

select '======================= selecting positive samples =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_geek_chat as
  select 
    uid geek_id, actionp boss_id, action
  from
    dw_bosszp.bg_action
  where
    ds > '$start_time' and ds <= '$end_time_plus'
  and
    action = 'chat'
  and
    bg = 0
  group by 
    uid, actionp, action;

select '======================= joining chat with add friend =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_boss_add_geek_chat as
  select 
    geek_id, job_id, action 
  from (
    select
      a.geek_id, a.job_id, 1 action
    from
      nlp.qyh_sample_boss_add a
    join
      nlp.qyh_sample_geek_chat b
    on
      a.geek_id = b.geek_id and a.boss_id = b.boss_id
  ) a
  group by
    a.geek_id, a.job_id, action;

select '======================= selecting add without chat =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_boss_add_no_chat as
  select
    a.geek_id, a.job_id, a.boss_id
  from
    nlp.qyh_sample_boss_add a
  left join
    nlp.qyh_sample_geek_chat b
  on
    a.geek_id = b.geek_id and a.boss_id = b.boss_id
  where
    b.action is null
  group by
    a.geek_id, a.job_id, a.boss_id;

select '======================= selecting active =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_active_geek as
  select
    uid geek_id
  from
    dw_bosszp.bg_action
  where
    ds > '$start_time' and ds <= '$end_time_plus'
  and
    action = 'list-boss'
  and
    bg = 0
  group by
    uid;

select '======================= selecting active and no chat =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_boss_add_no_chat_active as
  select
    a.geek_id, a.job_id, 0 action
  from
    nlp.qyh_sample_boss_add_no_chat a
  join
    nlp.qyh_sample_active_geek b
  on
    a.geek_id = b.geek_id;

select '======================= boss end =============================================' from dw_bosszp.zp_geek limit 1;

select '======================= joining samples =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample as
  select 
    geek_id, job_id, action 
  from
  (
    select 
      geek_id, job_id, action
      from 
        nlp.qyh_sample_geek_add_boss_chat
      distribute by rand() 
      sort by rand()
      limit $n_positive
    union all
    select
      geek_id, job_id, action
      from 
        nlp.qyh_sample_geek_add_no_chat_active
      distribute by rand() 
      sort by rand()
      limit $n_negative
    union all
    select
      geek_id, job_id, action
      from 
        nlp.qyh_sample_boss_add_geek_chat
      distribute by rand() 
      sort by rand()
      limit $n_positive
    union all
    select
      geek_id, job_id, action
      from 
        nlp.qyh_sample_boss_add_no_chat_active
      distribute by rand() 
      sort by rand()
      limit $n_negative  
  ) a
  group by
    geek_id, job_id, action;

insert overwrite table nlp.qyh_sample
  select
    geek_id, job_id, sum(action) action
  from
    nlp.qyh_sample
  group by
    geek_id, job_id;

create temporary table nlp.qyh_sample_filter as
  select 
    a.job_id, a.geek_id, a.action
  from 
    nlp.qyh_sample a
  join
    nlp.qyh_job b
  on
    a.job_id = b.job_id
  join
    nlp.qyh_resume c
  on
    a.geek_id = c.geek_id;

select '======================= joining profile =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_sample_with_profile as
select
  a.geek_id, 
  concat_ws('\t', 
    tokenizer(
      split(
        regexp_replace(
          concat_ws(' ', b.desc, b.prj, b.edu, b.workdesc), 
          '([？！，、；：“”‘（）《》〈〉【】『』「」﹃﹄〔〕…—～﹏￥]+)|(([0-9]|[一二三四五六七八九])[.、])',
          ' '
        ),
        '[。\t\n]+'
      )
    )
  ) geek_desc,
  a.job_id, 
  concat_ws('\t', 
    tokenizer(
      split(
        regexp_replace(
          c.job_desc,
          '([？！，、；：“”‘（）《》〈〉【】『』「」﹃﹄〔〕…—～﹏￥]+)|(([0-9]|[一二三四五六七八九])[.、])',
          ' '
        ),
        '[。\t\n]+'
      )
    )
  ) job_desc,  
  a.action
from
  nlp.qyh_sample_filter a
join
  nlp.qyh_resume b
on 
  a.geek_id = b.geek_id
join
  nlp.qyh_job c
on
  a.job_id = c.job_id;

select '======================= writing files =============================' from dw_bosszp.zp_geek limit 1;

create temporary table nlp.qyh_positive_sample_limit as
  select 
    geek_id, geek_desc, job_id, job_desc
  from
    nlp.qyh_sample_with_profile
  where action >= 1
  distribute by rand()
  sort by rand()
  limit $n_positive;

create temporary table nlp.qyh_negative_sample_limit as
  select
    geek_id, geek_desc, job_id, job_desc
  from
    nlp.qyh_sample_with_profile
  where action = 0
  distribute by rand()
  sort by rand()
  limit $n_negative;

insert overwrite local directory './negative'
  select * from nlp.qyh_positive_sample_limit;

insert overwrite local directory './positive'
  select * from nlp.qyh_negative_sample_limit;

insert overwrite local directory './all'
  select 
    geek_desc, job_desc
  from
    nlp.qyh_positive_sample_limit
  union all
  select
    geek_desc, job_desc
  from
    nlp.qyh_negative_sample_limit;
"

rm -r $fp
mkdir $fp
cat positive/* > $fp/$fp.positive
cat negative/* > $fp/$fp.negative
cat all/* > $fp/$fp.all
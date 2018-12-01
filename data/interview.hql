  insert overwrite table nlp.job_desc_token
    select
      id job_id,
      concat_ws('\t', 
        tokenizer(
          split(
            regexp_replace(
              concat_ws(
                '\t',
                collect_set(job_desc)
              ),
              '([，、；：“”‘（）《》〈〉【】『』「」﹃﹄〔〕…—～﹏￥]+)',
              ' '
            ),
            '([。？！\t\n]+)|(([0-9]|[一二三四五六七八九])[.、 ])'
          )
        )
      ) jd
    from
      dw_bosszp.zp_job
    group by
      id;  
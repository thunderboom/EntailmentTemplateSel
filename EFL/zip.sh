#!/bin/bash

project_name='../EFL/'
backup_name="EFL.zip"
all=""
for i in `find $project_name -name "*_output"`
do 
    all="$all $i/*"
done 
echo $all
# zip -r "$backup_name" "$project_name"  -x $all
tar czvf EFL.tar.gz $project_name --exclude=cluewsc_output --exclude=bustm_output --exclude=tnews_output \
                                  --exclude=chid_output --exclude=ocnli_output --exclude=iflytek_output \
                                  --exclude=eprstmt_output --exclude=csldcp_output --exclude=csl_output \
                                  --exclude=cmnli_output

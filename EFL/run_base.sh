# 1
for i in `seq 0 4`
do
  bash run_classifier_bustm_base.sh ${i} 
  wait
  bash run_classifier_bustm_base.sh ${i} predict
  wait
done
bash run_classifier_bustm_base.sh all 
wait
bash run_classifier_bustm_base.sh all predict
wait

# 2
for i in `seq 0 4`
do
    bash run_classifier_chid_base.sh $i 
    wait
    bash run_classifier_chid_base.sh $i predict
    wait
done
bash run_classifier_chid_base.sh all 
wait
bash run_classifier_chid_base.sh all predict
wait

# 3
for i in `seq 0 4`
do
	bash run_classifier_csl_base.sh $i 
    wait
    bash run_classifier_csl_base.sh $i predict
    wait
done
bash run_classifier_csl_base.sh all 
wait
bash run_classifier_csl_base.sh all predict
wait

# 4
for i in `seq 0 4`
do
	bash run_classifier_csldcp_base.sh $i 
    wait
    bash run_classifier_csldcp_base.sh $i predict
    wait
done
bash run_classifier_csldcp_base.sh all 
wait
bash run_classifier_csldcp_base.sh all predict
wait


# 5
for i in `seq 0 4`
do
	bash run_classifier_eprstmt_base.sh $i 
    wait
    bash run_classifier_eprstmt_base.sh $i predict
    wait
done
bash run_classifier_eprstmt_base.sh all 
wait
bash run_classifier_eprstmt_base.sh all predict
wait

# 6
for i in `seq 0 4`
do
	bash run_classifier_iflytek_base.sh $i 
    wait
    bash run_classifier_iflytek_base.sh $i predict
    wait
done
bash run_classifier_iflytek_base.sh all 
wait
bash run_classifier_iflytek_base.sh all predict
wait


# 7
for i in `seq 0 4`
do
	bash run_classifier_ocnli_base.sh $i 
    wait
    bash run_classifier_ocnli_base.sh $i predict
    wait
done
bash run_classifier_ocnli_base.sh all 
wait
bash run_classifier_ocnli_base.sh all predict
wait


# 8
for i in `seq 0 4`
do
	bash run_classifier_tnews_base.sh $i 
    wait
    bash run_classifier_tnews_base.sh $i predict
    wait
done
bash run_classifier_tnews_base.sh all 
wait
bash run_classifier_tnews_base.sh all predict
wait


# 9
for i in `seq 0 4`
do
	bash run_classifier_wsc_base.sh $i 
    wait
    bash run_classifier_wsc_base.sh $i predict
    wait
done
bash run_classifier_wsc_base.sh all 
wait
bash run_classifier_wsc_base.sh all predict
wait

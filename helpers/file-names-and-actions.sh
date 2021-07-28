cd ./hmdb51_org_subset/
lines=""
for folder in *; do
    cd $folder
    for file in *.avi; do
        if [ "${lines}" == "" ]
        then
            lines="${file} ${folder}"
        else
            lines="${lines}"$'\r'"${file} ${folder}"
        fi
    done
    cd ..
done
echo $lines > ../data/hmdb/split0_master.txt
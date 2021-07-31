cd ./hmdb51_org_subset/
for folder in *; do
    cd $folder
    for file in *.avi; do
        ffmpeg -i $file -c:v copy -c:a copy "${file%.*}_0.mp4" -hide_banner -loglevel error
        mv "${file%.*}_0.mp4" ../../ssd/hmdb/"${file%.*}_0.mp4"
    done
    cd ..
done
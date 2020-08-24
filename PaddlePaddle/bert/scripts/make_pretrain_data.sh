echo "gzip demo_wiki_train.gz ..."
gzip -d demo_wiki_train.gz
echo "Done!"

for i in {1..50}                  
do
  cat demo_wiki_train >> demo_wiki_train_50        
  echo "Copy $i times"
  let i+=1
done

echo "gzip demo_wiki_train_50 to demo_wiki_train_50.gz ..."
gzip -k demo_wiki_train_50
echo "Success!"

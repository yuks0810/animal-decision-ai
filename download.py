from flickrapi import FlickrAPI # flickrのAPIを使う
from urllib.request import urlretrieve # クローリングかな？わかんない
from pprint import pprint # データを表示するためのもの
import os, time, sys # os情報を取得

# APIキーの情報
key = "d1c139827feaa05f87fb01d0bd1b4574"
secret = "545553e666ee0040"
wait_time = 1 # リクエストを１秒毎に（攻撃とみなされるかもだから）

# 保存フォルダ
animalname = sys.argv[1] #コマンドラインの入力情報の２番目のパラメータを指す
savedir = "./" + animalname # カレントディレクトリのanimalnameというファイルに保存

flickr = FlickrAPI(key, secret, format='parsed-json')
result = flickr.photos.search(
    text = animalname,
    per_page = 400, # 400このデータ
    media = 'photos',
    sort = 'relevance',
    safe_search = 1,
    extras = 'url_q, licence' #取得したい情報（写真とライセンス）
)

photos = result['photos']
# pprint(photos)

# ここから差分（１）---------------------------------------------------------------
for i, photo in enumerate(photos['photo']):
    url_q = photo['url_q']
    filepath = savedir + '/' + photo['id'] + '.jpg'
    # 重複していなければ保存、していればスキップ
    if os.path.exists(filepath): continue
    urlretrieve(url_q, filepath)
    time.sleep(wait_time)

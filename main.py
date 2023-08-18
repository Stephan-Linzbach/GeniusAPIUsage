# This is a sample Python script.
import lyricsgenius
import re
import random
import json
from collections import defaultdict

# Press Umschalt+F10 to execute it or replace it with your code.
#"Vega": 0,"amar ": 0,
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.,

with open("rapper_id.json") as f:
    rappers = json.load(f)


genius = lyricsgenius.Genius("a9u8Dj51CZKQmNHn0k5C7OLUtS6ak-YzUgLlK5k2RoM5JjNn84YGJXWokP45ltFY")


def get_rapper_id(name):
    return genius.search_artist(name, max_songs=10, sort='popularity', include_features=False)


def get_part(lyric, start, end):
    return lyric[start:end]


def parse_dict(occurences):
    description = range(0, len(occurences), 2)
    parts = range(1, len(occurences), 2)
    return {occurences[d] : [l for l in occurences[p].split('\n') if l] for d, p in zip(description, parts)}


# Press the green button in the gutter to run the script.

def random_shuffle(a):
    keys = list(a.keys())
    random.shuffle(keys)
    b = {k : a[k] for k in keys}
    return b


def get_songs_for_id(id):
    page = 1
    songs = []
    while page:
        request = genius.search_artist(id,
                                      sort='popularity',
                                      per_page=50,
                                      page=page)
        songs.append(request['songs'])
        page = request['next_page']
    return songs


def get_lyric(i):
    print(i)
    return genius.lyrics(i)


def get_rapper_ids(rappers):
    for n, id in rappers.keys():
        found_artists = genius.search(n, type_='artist')
        print(found_artists)
        if not found_artists:
            print(n)
            continue
        all_artists = [f['result'] for f in found_artists['sections'][0]['hits']]
        for a in all_artists:
            try:
                name = a['name'].lower()
                if name[0] == " ":
                    name = name[1:]
                if n == name:
                    rappers[n] = a['id']
                    break
            except:
                pass
    return rappers


if __name__ == '__main__':
    rapper = list(rappers.keys())
    parts = defaultdict(list)
    ignore =['kc rebell']
    """ignore = ['kc rebell',
              'casper',
              'tua',
              'prinz pi',
              'marteria',
              "brutos brutaloz",
              "beslik meister",
              'negatiiv og',
              'toobrokeforfiji',
              'sevi rin',
              'rin',
              'makko (deu)', # TDOO
              'sin davis',
              'haftbefehl',
              'sido',
              'bushido',
              'shindy',
              'ak ausserkontrolle',
              'teesy',
              'kaas',
              'megaloh',
              'cro',
              'money boy',
              'hustensaft jüngling',
              'medikamenten manfred',
              'bartek',
              'yin kalle',
              'symba',
              'kasimir1441',
              'ufo361',
              'kraftklub',
              'eins zwo',
              'fettes brot',
              'gzuz',
              'bonez mc',
              'raf camora',
              'nmzs',
              'koljah',
              'panik panzer',
              'eko fresh',
              'kool savas',
              'summer cem',
              'farid bang', #TODO
              'kollegah',
              'chakuza',
              'skeeniboi',
              'baba saad',
              'soufian',
              'can mit me$$r',
              'mero',
              'capital bra',
              'samra',
              'og keemo',
              'pashanim',
              #'fler', #TODO
              'prinz porno',
              'sierra kidd',
              'edo saiya',
              'lakmann one', #TODO
              'al kareem',
              ]"""
    for n, id in rappers.items():
        if n in ignore:
            continue
        try:
            artist = genius.search_artist(n, max_songs=120, sort='popularity', per_page=50, artist_id=id)
            songs = [s for s in artist.songs]
            lyrics = [l.lyrics for l in songs]
            for l in lyrics:
                print("for loop")
                #print(l)
                parts[n].append(l)
                """occurences = re.split(r'\[(.*?)\]', l)[1:]
                print(lyric_dict)
                for k, v in lyric_dict.items():
                    if ':' in k:
                        real_rapper = [r for r in rapper if r in k.lower()]
                        if not len(real_rapper) == 0:
                            parts[real_rapper[0]].append(v)
                    else:
                        parts[n].append(v)"""
            with open(n+".json", "w") as f:
                json.dump(dict(parts), f)
        except Exception as e:
            pass
    """with open("dlkas.json", "w") as f:
        json.dump(dict(parts), f)"""




# See PyCharm help at https://www.jetbrains.com/help/pycharm/

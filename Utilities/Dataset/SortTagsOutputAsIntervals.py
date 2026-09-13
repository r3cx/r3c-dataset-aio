string1 = "melon22, kaedeno yuu, cromachina, derauea, funnyari, gemba \(dlfms75\), tedain, thanabis, tottotonero, trente, chirumakuro, yamaori, kz oji, sunhyun, 5danny1206, ayagi daifuku, camonome, cream cod, diisuke, free style \(yohan1754\), hotate-chan, hungry clicker, komusou \(jinrikisha\), monaka \(gatinemiku\), oekakizuki, onono imoko, poper \(arin sel\), qian wu atai, shuz \(dodidu\), thomasz, toudori, wagashi928, wagashi \(dagashiya\), yaegashi nan, aikome \(haikome\), alexmaster, belko, chamchami, drunkoak, hamu 767, hydrant \(kasozama\), laserflip, m-da s-tarou, neneneji, nez-box, ru zhai, shiokonbu, u ronnta, umigarasu \(kitsune1963\), uo denim, yabai gorilla, sanshoku amido, serin199, snowball22, suurin \(ksyaro\), thirty 8ght, uenomigi, waterring, zer0.zer0, basukechi, chipa \(arutana\), dev voxy, powzin, ppolar, 9is, alexi \(tits!\), b-ginga, clarevoir, dikko, greem bang, haoni, janong, kaptivate, kumasteam, mikozin, mizumizuni, moisture \(chichi\), osiimi, banana oekaki, greatmosu, konno tohiro, niliu chahui"

string2 = "asanagi, toosaka asagi, pentagon \(railgun ky1206\), oryo \(oryo04\), fujima takuya, kani biimu, ebifurya, yd \(orange maru\), sincos, riichu, lack, watanabe akio, clearite, morikura en, rangu, parsley-f, hews, syhan, dokomon, yanyo \(ogino atsuki\), hiroki \(yyqw7151\), nel-zel_formula, ikuchan_kaoru, jjune, shin'ya \(shin'yanchi\), greem bang, sumiyao \(amam\), hidulume, sak \(lemondisk\), takayaki, thomas 8000, ndgd, deyui, deadnoodles, hagoonha, quan \(kurisu tina\), sheya, harada \(sansei rain\), hiroikara \(smhong04\), enru, gugu0v0, reia, ke-ta, kantoku, na-ga, nakta, pan \(mimi\), eluthel, mochigome \(ununquadium\), kgt \(pixiv12957613\), hara yui, reoen, rella, atdan, hiten \(hitenkei\), gomzi, free style \(yohan1754\), modare, qiandaiyiyu, fuzichoco, john_kafka, sakatsuki yakumo, dino \(dinoartforame\), haoni, oekakizuki, ohisashiburi, mochirong, merrytail, kairunoburogu, niliu chahui, mikozin, pottsness, gweda, ishikei, hero neisan, mignon, quasarcake, ciloranko, freng, nyantcha, atahuta, lam \(ramdayo\), mika pikazo, kaamin \(mariarose753\), zankuro, opossumachine, kakure eria, quasarcake, kazukoto, uenomigi, baffu, ciloranko, tianliang duohe fangdongye, mizumizuni, kashu \(hizake\), georugu13, sakamata \(sakamata4\), shiba \(zudha\), zankuro, shiokonbu, atte nanakusa, kenkou cross, torisan, ru zhai, fumihiko \(fu mihi ko\), tamada heijun, lam \(ramdayo\), betabeet, qiandaiyiyu, shibori kasu, chomikuplus, tenchisouha"

string = string1#+", "+string2

interval = 10

lst = string.split(',')
lst = [x.strip() for x in set(lst)]
lst.sort()

output = []
temp = []

for i, s in enumerate(lst):
    if (i+1)%interval == 0:
        output.append(', '.join(temp))
        temp = list()
        temp.append(s)
    else:
        temp.append(s)

if len(temp):
    output.append(', '.join(temp))

print()
[print(x) for x in output]

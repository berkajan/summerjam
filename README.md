# Smart Beehive
Github repositář pro soutěž CRa Summerjam 2018

# O co jde

Smart Beehive je webová aplikace pro sledování a chytrou správu včelstev, díky LoRa senzoru hmotnosti, teploty, vlhkosti a atmosferického tlaku.
Řešení funguje tak, že:

1) přes REST API sbírá data ze senzoru, 
2) získaná data zpracovává (průměruje, detekuje outliery, trénuje prediktivní model hmostnosti, ...),
3) poskytuje vizualizaci ve webovém prohlížeči (grafy vývoje historických dat, aktuální stav, predikci, tabulku událostí, doporučené akce...).

# Seznam použitých technologií
LoRa WAN senzor hmotnosti Beepad (id senzoru 0004A30B001F216B)
Python 3 s knihovnami:
requests
pandas
numpy
datetime
sklearn
bokeh

Python aplikace běží na Linux serveru (nicméně Linux není nutnost, stačí python 3 a příslušné knihovny), otestováno v Ubuntu 14.04.

http://80.211.113.38:5006/bees

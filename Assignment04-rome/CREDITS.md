# Image credits


## Concept diagrams

All diagram PNGs were drawn for this notebook with matplotlib by the course team's build scripts. They are part of
the course material.

route.png, visual_words.png, retrieval_vs_all_pairs.png, query_expansion.png, connected_components.png,
maximum_spanning_tree.png, similarity_alignment.png, rome_scale.png: drawn for this notebook.

They are drawn as atlas plates in the manner of the Piranesi and Nolli illustrations below. The lettering uses two
fonts under the SIL Open Font License 1.1 (https://openfontlicense.org), downloaded from the Google Fonts repository
(https://github.com/google/fonts, `ofl/cinzel` and `ofl/ebgaramond`), with their licence files; static weights are
cut from the variable fonts. The PNGs are rasterised, so the fonts are not needed to view them. Maths is set in STIX
(bundled with matplotlib).

| Font | Use | Copyright | Licence |
|---|---|---|---|
| Cinzel (Cinzel[wght].ttf, weights 400 and 600) | plate titles, TAV. numbers, capitals | Copyright 2020 The Cinzel Project Authors (https://github.com/NDISCOVER/Cinzel) | SIL OFL 1.1 |
| EB Garamond (EBGaramond[wght].ttf, EBGaramond-Italic[wght].ttf; weights 400 and 500, italic 400) | labels, captions, numbers | Copyright 2017 The EB Garamond Project Authors (https://github.com/octaviopardo/EBGaramond12) | SIL OFL 1.1 |

The numbers about Building Rome in a Day in `rome_scale.png` (150,000 images, 496 cores on 62 computers, 13 h
matching + 1 h skeletal sets + 7 h SfM, 8,825,256 pairs verified, 2,712,301 found, 36,658 registered) are from
Agarwal et al., "Building Rome in a Day", Communications of the ACM 54(10), 2011, Table 1. The times in
`retrieval_vs_all_pairs.png` use the notebook's illustration rate of 2 ms per pair check.

## Illustrations (public domain, Wikimedia Commons)

Etchings by Giovanni Battista Piranesi (1720 to 1778) from the *Vedute di Roma* (published 1748 to 1778), scanned by
the Statens Museum for Kunst (SMK, Copenhagen), and the reduced version of Giovanni Battista Nolli's 1748 plan of Rome
(engraved by Piranesi and Nolli), scanned by the Bibliotheque nationale de France. Each file is marked Public domain on
its Commons description page (LicenseShortName "Public domain" in the Commons API extmetadata; the artists died more
than 200 years ago). Licences checked on 2026-10-08. The files here are greyscale copies of the Commons thumbnails
(960 px), trimmed to the printed plate, resized to at most 780 px wide (600 px in the student repository) and
recompressed (the download script refuses any file that is not marked Public domain). Dates are as given on Commons
(the SMK impression dates).

| File | Title (caption on the plate) | Artist | Date | Source | Licence on Commons |
|---|---|---|---|---|---|
| plate_title_frontispiece.jpg | Frontispiece of the *Vedute di Roma*, with the statue of Minerva among ruins (SMK title: "Udsigter fra Rom. Frontispiece") | Giovanni Battista Piranesi | 1748 | https://commons.wikimedia.org/wiki/File:Giovanni_Battista_Piranesi,_Udsigter_fra_Rom_Frontispiece,_1748,_KKSgb9860-2,_Statens_Museum_for_Kunst.jpg | Public domain |
| plate_castel_sant_angelo.jpg | Veduta del Ponte e Castello Sant'Angelo (SMK title: "Castel Sant' Angelo-broen med kastellet (Hadrians mausoleum) ...") | Giovanni Battista Piranesi | 1754 | https://commons.wikimedia.org/wiki/File:Giovanni_Battista_Piranesi,_Castel_Sant%27_Angelo-broen_med_kastellet_Hadrians_mausoleum_,_i_forgrunden_Tiberen_med_b%C3%A5de_og_i_det_fjerne_Peterskirkens_kuppel,_1754,_KKSgb9860-29,_Statens_Museum_for_Kunst.jpg | Public domain |
| plate_pantheon.jpg | Veduta del Pantheon d'Agrippa, oggi Chiesa di S. Maria ad Martyres (SMK title: "Pantheon") | Giovanni Battista Piranesi | 1761 | https://commons.wikimedia.org/wiki/File:Giovanni_Battista_Piranesi,_Pantheon,_1761,_KKSgb9860-60,_Statens_Museum_for_Kunst.jpg | Public domain |
| plate_forum.jpg | Veduta di Campo Vaccino (the Roman Forum; SMK title: "Forum Romanum, eller Campo Vaccino ...") | Giovanni Battista Piranesi | 1772 | https://commons.wikimedia.org/wiki/File:Giovanni_Battista_Piranesi,_Forum_Romanum,_eller_Campo_Vaccino,_Antoninus%27_og_Faustina%27s_tempel_til_venstre,_templet_for_Castor_og_Pollux_til_h%C3%B8jre_og_Colosseum_i_det_fjerne,_1772,_KKSgb9860-100,_Statens_Museum_for_Kunst.jpg | Public domain |
| map_nolli_1748.jpg | La Topografia di Roma di Gio. Battista Nolli, dalla maggiore in questa minor tavola dal medesimo ridotta. Piranesi e Nolli incisero (the small Nolli plan) | Giovanni Battista Nolli, engraved with Giovanni Battista Piranesi | 1748 | https://commons.wikimedia.org/wiki/File:La_Topografia_di_Roma_di_-_Gio._Battista_Nolli,_dalla_maggiore_in_questa_minor_tavola_dal_medesimo_ridotta._Piranesi_e_Nolli_incisero_-_btv1b53064508x.jpg | Public domain |
| plate_trevi.jpg | Veduta in prospettiva della gran Fontana dell'Acqua Vergine detta di Trevi. Architettura di Nicola Salvi (SMK title: "Trevifontaenen") | Giovanni Battista Piranesi | 1773 | https://commons.wikimedia.org/wiki/File:Giovanni_Battista_Piranesi,_Trevifont%C3%A6nen,_1773,_KKSgb9860-104,_Statens_Museum_for_Kunst.jpg | Public domain |
| plate_st_peters.jpg | Veduta della gran Piazza e Basilica di S. Pietro situata ove erano anticamente il Circo e gl'Orti di Cajo e Nerone nella Valle Vaticana (SMK title: "Peterspladsen") | Giovanni Battista Piranesi | 1772 | https://commons.wikimedia.org/wiki/File:Giovanni_Battista_Piranesi,_Peterspladsen,_1772,_KKSgb9860-101,_Statens_Museum_for_Kunst.jpg | Public domain |
| plate_colosseum.jpg | Veduta dell'Anfiteatro Flavio detto il Colosseo (SMK title: "Colosseum") | Giovanni Battista Piranesi | 1760 | https://commons.wikimedia.org/wiki/File:Giovanni_Battista_Piranesi,_Colosseum,_1760,_KKSgb9860-57,_Statens_Museum_for_Kunst.jpg | Public domain |
| plate_arch_of_constantine.jpg | Veduta dell'Arco di Costantino (SMK title: "Constantin's triumfbue") | Giovanni Battista Piranesi | 1771 | https://commons.wikimedia.org/wiki/File:Giovanni_Battista_Piranesi,_Constantin%27s_triumfbue,_1771,_KKSgb9860-97,_Statens_Museum_for_Kunst.jpg | Public domain |

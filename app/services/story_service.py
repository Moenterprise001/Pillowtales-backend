from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, List

import google.generativeai as genai
from fastapi import HTTPException

from app.core.config import settings
from app.domain.constants import STORY_COMPANIONS, SUBSCRIPTION_TIERS, SUPPORTED_LANGUAGES
from app.models.story import GenerateStoryRequest
from app.models.subscription import SubscriptionResponse
from app.repositories.story_repository import StoryRepository
from app.utils.story_text import postprocess_story_pages


OPENING_SEED_FAMILIES = [
    {
        "family": "sleepy_forest_path",
        "en": "{childName} lived near the edge of a sleepy forest, where a mossy path curled between soft ferns and quiet trees.",
        "es": "Al borde del bosque tranquilo detrás de la casa de {childName}, una pequeña linterna empezó a brillar junto a un sendero cubierto de musgo.",
        "fr": "À l’orée de la forêt endormie derrière la maison de {childName}, une petite lanterne se mit à briller près d’un sentier moussu.",
        "it": "Ai margini del bosco addormentato dietro casa di {childName}, una piccola lanterna cominciò a brillare accanto a un sentiero di muschio.",
        "de": "Am Rand des schläfrigen Waldes hinter {childName}s Zuhause begann neben einem moosigen Pfad eine kleine Laterne zu leuchten.",
    },
    {
        "family": "lantern_treehouse",
        "en": "{childName} loved spending evenings inside a warm treehouse lit by tiny lanterns, where the wooden walls creaked softly in the breeze.",
        "es": "Dentro de una casita en un árbol iluminada por farolillos cálidos, {childName} encontró un pequeño cartel de madera que señalaba hacia las estrellas.",
        "fr": "Dans une cabane perchée toute chaude, éclairée par de petites lanternes, {childName} trouva un panneau de bois qui indiquait les étoiles.",
        "it": "Dentro una casetta sull’albero illuminata da piccole lanterne calde, {childName} trovò un cartello di legno che indicava le stelle.",
        "de": "In einem warmen Baumhaus voller kleiner Laternen fand {childName} ein hölzernes Schild, das zu den Sternen zeigte.",
    },
    {
        "family": "hidden_garden_gate",
        "en": "{childName} loved helping in the little garden beside home, where every flower seemed to have its own secret.",
        "es": "Más allá de la verja tranquila del jardín, {childName} entró en un lugar iluminado por la luna donde cada flor parecía a punto de susurrar hola.",
        "fr": "Au-delà du petit portail du jardin, {childName} entra dans un lieu baigné de lune où chaque fleur semblait prête à murmurer bonsoir.",
        "it": "Oltre il cancelletto silenzioso del giardino, {childName} entrò in un luogo illuminato dalla luna dove ogni fiore sembrava pronto a salutare piano.",
        "de": "Hinter dem stillen Gartentor trat {childName} in einen mondhellen Ort, an dem jede Blume leise Hallo sagen wollte.",
    },
    {
        "family": "cloud_island",
        "en": "{childName} spent many evenings watching soft clouds drift above the rooftops, imagining little islands hidden in the sky.",
        "es": "En una suave isla de nubes sobre los tejados, {childName} vio un puente plateado que se curvaba despacio hacia la noche.",
        "fr": "Sur une douce île de nuages au-dessus des toits, {childName} aperçut un pont argenté qui s’arrondissait dans la nuit.",
        "it": "Su una morbida isola di nuvole sopra i tetti, {childName} vide un ponte d’argento che curvava piano nella notte.",
        "de": "Auf einer weichen Wolkeninsel über den Dächern entdeckte {childName} eine silberne Brücke, die sich sanft in die Nacht bog.",
    },
    {
        "family": "moonlit_library",
        "en": "{childName} loved listening to stories in a quiet moonlit library, where the books rested on warm wooden shelves like sleepy friends.",
        "es": "En una biblioteca tranquila iluminada por la luna, donde los libros parecían respirar despacito, {childName} vio que un cuento se abría solo.",
        "fr": "Dans une bibliothèque calme éclairée par la lune, où les livres semblaient respirer tout doucement, {childName} vit un album s’ouvrir tout seul.",
        "it": "In una biblioteca silenziosa illuminata dalla luna, dove i libri sembravano respirare piano, {childName} vide un libro aprirsi da solo.",
        "de": "In einer stillen Mondlicht-Bibliothek, in der die Bücher leise zu atmen schienen, sah {childName}, wie sich ein Geschichtenbuch von selbst öffnete.",
    },
    {
        "family": "seaside_cave",
        "en": "{childName} lived near a cosy little cave beside the calm sea, where the waves hummed softly against the shore each evening.",
        "es": "Dentro de una pequeña cueva acogedora junto al mar en calma, {childName} oyó cómo las olas tarareaban una nana.",
        "fr": "Dans une petite grotte douillette au bord de la mer calme, {childName} entendit les vagues fredonner une berceuse.",
        "it": "Dentro una piccola grotta accogliente vicino al mare calmo, {childName} sentì le onde canticchiare una ninna nanna.",
        "de": "In einer gemütlichen kleinen Höhle am ruhigen Meer hörte {childName}, wie die Wellen ein Schlaflied summten.",
    },
    {
        "family": "sleepy_castle_hall",
        "en": "{childName} lived in a sleepy castle with long quiet hallways, where glowing pictures watched kindly from the walls.",
        "es": "Por un pasillo de castillo adormecido, lleno de cuadros brillantes, {childName} encontró una puerta dorada entreabierta.",
        "fr": "Dans un couloir de château endormi bordé de tableaux lumineux, {childName} trouva une porte dorée entrouverte.",
        "it": "Lungo un corridoio di castello addormentato, pieno di quadri luminosi, {childName} trovò una porta dorata socchiusa.",
        "de": "In einem schläfrigen Schlossflur voller leuchtender Bilder fand {childName} eine goldene Tür, die einen Spalt offen stand.",
    },
    {
        "family": "glowing_attic",
        "en": "{childName} often wondered what was hidden in the little attic above the stairs, where dusty boxes and old stories waited quietly.",
        "es": "En el pequeño desván sobre las escaleras, {childName} descubrió un mapa antiguo que brillaba suavemente sobre las tablas del suelo.",
        "fr": "Dans le petit grenier au-dessus de l’escalier, {childName} découvrit une vieille carte qui brillait doucement sur le plancher.",
        "it": "Nella piccola soffitta sopra le scale, {childName} scoprì una vecchia mappa che brillava piano sulle assi del pavimento.",
        "de": "Auf dem kleinen Dachboden über der Treppe entdeckte {childName} eine alte Karte, die sanft auf den Dielen leuchtete.",
    },
    {
        "family": "snowy_village",
        "en": "{childName} lived in a tiny snowy village tucked beneath the stars, where warm windows glowed on cosy winter evenings.",
        "es": "En una pequeña aldea nevada bajo las estrellas, {childName} vio ventanas cálidas que parpadeaban como ojos amables.",
        "fr": "Dans un minuscule village enneigé blotti sous les étoiles, {childName} vit des fenêtres chaudes cligner comme des yeux gentils.",
        "it": "In un piccolo villaggio innevato sotto le stelle, {childName} vide finestre calde lampeggiare come occhi gentili.",
        "de": "In einem winzigen verschneiten Dorf unter den Sternen sah {childName} warme Fenster wie freundliche Augen blinzeln.",
    },
    {
        "family": "river_of_stars",
        "en": "{childName} lived beside a slow river that reflected every star, where the water moved as gently as a bedtime song.",
        "es": "Junto a un río lento que reflejaba todas las estrellas, {childName} encontró un barquito de papel esperando en la orilla.",
        "fr": "Au bord d’une rivière lente qui reflétait toutes les étoiles, {childName} trouva un petit bateau de papier qui attendait sur la rive.",
        "it": "Accanto a un fiume lento che rifletteva tutte le stelle, {childName} trovò una barchetta di carta in attesa sulla riva.",
        "de": "Neben einem langsamen Fluss, der jeden Stern spiegelte, fand {childName} ein kleines Papierboot am Ufer warten.",
    },
    {
        "family": "meadow_clock",
        "en": "{childName} loved walking through a quiet meadow where the grass shone silver and bluebells nodded softly in the evening breeze.",
        "es": "En un prado tranquilo donde la hierba brillaba plateada, {childName} vio un diminuto reloj haciendo tic-tac dentro de una campanilla azul.",
        "fr": "Dans une prairie calme où l’herbe brillait d’argent, {childName} remarqua une minuscule horloge qui tic-taquait dans une jacinthe des bois.",
        "it": "In un prato silenzioso dove l’erba brillava d’argento, {childName} notò un minuscolo orologio che ticchettava dentro una campanula blu.",
        "de": "Auf einer stillen Wiese, auf der das Gras silbern schimmerte, bemerkte {childName} eine winzige Uhr, die in einer Glockenblume tickte.",
    },
    {
        "family": "pillow_harbour",
        "en": "{childName} built a tiny harbour from pillows and blankets, where pretend boats rested safely before bedtime.",
        "es": "En un pequeño puerto hecho de almohadas y mantas, {childName} encontró una barquita lunar balanceándose junto a la cama.",
        "fr": "Dans un petit port fait d’oreillers et de couvertures, {childName} trouva une barque de lune qui flottait près du lit.",
        "it": "In un piccolo porto fatto di cuscini e coperte, {childName} trovò una barchetta di luna che dondolava accanto al letto.",
        "de": "In einem kleinen Hafen aus Kissen und Decken fand {childName} ein Mondboot, das neben dem Bett schaukelte.",
    },
    {
        "family": "amazon_treehouse",
        "en": "{childName} lived high in a rainforest treehouse wrapped in green leaves and warm lantern light, where colourful birds greeted each morning.",
        "es": "En lo alto de una casita en un árbol de la selva, rodeada de hojas verdes y farolillos cálidos, {childName} oyó a una mariposa dorada tocar suavemente la barandilla de madera.",
        "fr": "Très haut dans une cabane de forêt tropicale entourée de feuilles vertes et de petites lanternes, {childName} entendit un papillon doré tapoter doucement la rambarde de bois.",
        "it": "In alto, in una casa sull’albero nella foresta tropicale, tra foglie verdi e piccole lanterne calde, {childName} sentì una farfalla dorata bussare piano alla ringhiera di legno.",
        "de": "Hoch oben in einem Regenwald-Baumhaus zwischen grünen Blättern und warmem Laternenlicht hörte {childName}, wie ein goldener Schmetterling leise an das Holzgeländer klopfte.",
    },
    {
        "family": "nile_river_boat",
        "en": "{childName} travelled with family on a quiet little boat drifting along a wide moonlit river, where the water shimmered softly beneath the stars.",
        "es": "En una barquita tranquila que avanzaba por un río ancho iluminado por la luna, {childName} vio unos juncos antiguos inclinarse como si señalaran el camino.",
        "fr": "Sur une petite barque tranquille qui glissait sur un large fleuve éclairé par la lune, {childName} vit de grands roseaux anciens se pencher comme s’ils montraient le chemin.",
        "it": "Su una piccola barca tranquilla che scivolava lungo un grande fiume illuminato dalla luna, {childName} vide antiche canne piegarsi come per indicare la strada.",
        "de": "Auf einem kleinen ruhigen Boot, das über einen breiten mondhellen Fluss glitt, sah {childName} alte Schilfhalme, die sich beugten, als wollten sie den Weg zeigen.",
    },
    {
        "family": "desert_caravan",
        "en": "{childName} travelled with a gentle desert caravan that rested under violet stars, where the sand felt cool and quiet at night.",
        "es": "Junto a una tranquila caravana del desierto que descansaba bajo estrellas violetas, {childName} encontró una piedra azul y fresca que zumbaba suavemente en la arena.",
        "fr": "Près d’une douce caravane du désert arrêtée sous des étoiles violettes, {childName} trouva un galet bleu et frais qui bourdonnait doucement dans le sable.",
        "it": "Accanto a una tranquilla carovana del deserto ferma sotto stelle violette, {childName} trovò un sassolino blu e fresco che vibrava piano nella sabbia.",
        "de": "Neben einer sanften Wüstenkarawane, die unter violetten Sternen ruhte, fand {childName} einen kühlen blauen Kiesel, der leise im Sand summte.",
    },
    {
        "family": "underwater_palace",
        "en": "{childName} lived in a glowing underwater palace where silver fish swam through archways and gentle lights shimmered through the water each evening.",
        "es": "En un palacio submarino brillante, donde peces plateados pasaban bajo los arcos, {childName} vio abrirse una puerta de perla sin hacer ruido.",
        "fr": "Dans un palais sous-marin lumineux où des poissons argentés passaient sous les arches, {childName} vit une porte de nacre s’ouvrir sans bruit.",
        "it": "In un palazzo sottomarino luminoso, dove pesci d’argento nuotavano tra gli archi, {childName} vide una porta di perla aprirsi senza rumore.",
        "de": "In einem leuchtenden Unterwasserpalast, in dem silberne Fische durch Bögen schwammen, sah {childName}, wie sich eine Perlentür lautlos öffnete.",
    },
    {
        "family": "pirate_harbour",
        "en": "{childName} lived in a peaceful pirate harbour where lanterns glowed on sleepy ships and sailors shared stories beneath the stars.",
        "es": "En un puerto pirata tranquilo, lleno de velas dormidas y faroles, {childName} descubrió una pequeña brújula que giraba hacia una isla secreta.",
        "fr": "Dans un port de pirates paisible, rempli de voiles endormies et de lanternes, {childName} découvrit une petite boussole qui tournait vers une île secrète.",
        "it": "In un porto di pirati tranquillo, pieno di vele addormentate e lanterne, {childName} scoprì una piccola bussola che girava verso un’isola segreta.",
        "de": "In einem friedlichen Piratenhafen voller schläfriger Segel und Laternen entdeckte {childName} einen kleinen Kompass, der zu einer geheimen Insel zeigte.",
    },
    {
        "family": "dragon_market",
        "en": "{childName} spent happy days in a cosy mountain market where gentle dragons warmed cups of cocoa and everyone knew one another by name.",
        "es": "En un mercado acogedor de montaña, donde unos dragones tranquilos calentaban tazas de cacao con pequeños soplidos, {childName} vio un puesto que brillaba más que los demás.",
        "fr": "Dans un marché douillet de montagne où de doux dragons réchauffaient des tasses de chocolat avec de petits souffles, {childName} remarqua une échoppe plus lumineuse que les autres.",
        "it": "In un accogliente mercato di montagna, dove draghi gentili scaldavano tazze di cacao con piccoli sbuffi, {childName} notò una bancarella più luminosa delle altre.",
        "de": "Auf einem gemütlichen Bergmarkt, wo sanfte Drachen mit kleinen Pustern Kakaotassen wärmten, bemerkte {childName} einen Stand, der heller leuchtete als alle anderen.",
    },
    {
        "family": "crystal_cavern",
        "en": "{childName} lived near a quiet crystal cavern, where the walls chimed softly whenever moonlight touched the stone.",
        "es": "En lo más profundo de una cueva tranquila de cristal, {childName} oyó cómo las paredes tintineaban suavemente mientras un camino de chispas azules aparecía en el suelo.",
        "fr": "Au fond d’une grotte de cristal silencieuse, {childName} entendit les parois tinter doucement tandis qu’un chemin d’étincelles bleues apparaissait au sol.",
        "it": "Nel profondo di una silenziosa caverna di cristallo, {childName} sentì le pareti tintinnare piano mentre una scia di scintille blu appariva sul pavimento.",
        "de": "Tief in einer stillen Kristallhöhle hörte {childName}, wie die Wände leise klangen, während eine Spur blauer Funken auf dem Boden erschien.",
    },
    {
        "family": "northern_lights_village",
        "en": "{childName} lived in a tiny village beneath the northern lights, where ribbons of colour danced gently across the sky.",
        "es": "En una pequeña aldea bajo la aurora boreal, {childName} vio cintas de colores bajar del cielo y señalar un sendero nevado.",
        "fr": "Dans un petit village sous les aurores boréales, {childName} vit des rubans de couleur descendre du ciel et montrer un sentier enneigé.",
        "it": "In un piccolo villaggio sotto l’aurora boreale, {childName} vide nastri di colore scendere dal cielo e indicare un sentiero innevato.",
        "de": "In einem kleinen Dorf unter dem Nordlicht sah {childName}, wie bunte Bänder vom Himmel herabglitten und auf einen verschneiten Pfad zeigten.",
    },
    {
        "family": "sky_train_station",
        "en": "{childName} lived near a little sky-train station floating between soft clouds, where gentle bells rang when the stars came out.",
        "es": "En una pequeña estación de trenes del cielo, flotando entre nubes suaves, {childName} encontró un billete dorado con el amanecer de mañana pintado encima.",
        "fr": "Dans une petite gare du ciel flottant entre de doux nuages, {childName} trouva un billet doré où était peint le lever du soleil du lendemain.",
        "it": "In una piccola stazione dei treni del cielo, sospesa tra nuvole morbide, {childName} trovò un biglietto dorato con dipinta l’alba del giorno dopo.",
        "de": "An einem kleinen Himmelsbahnhof zwischen weichen Wolken fand {childName} eine goldene Fahrkarte, auf die der Sonnenaufgang von morgen gemalt war.",
    },
    {
        "family": "jungle_waterfall",
        "en": "{childName} lived beside a gentle jungle waterfall where colourful birds greeted each morning and silver leaves danced in the moonlight.",
        "es": "Junto a una cascada tranquila de la selva que brillaba a la luz de la luna, {childName} descubrió una puerta redonda de piedra escondida detrás de hojas plateadas.",
        "fr": "Près d’une douce cascade de jungle qui brillait au clair de lune, {childName} aperçut une porte ronde en pierre cachée derrière des feuilles argentées.",
        "it": "Accanto a una tranquilla cascata nella giungla che brillava alla luce della luna, {childName} vide una porta rotonda di pietra nascosta dietro foglie d’argento.",
        "de": "Neben einem sanften Dschungelwasserfall, der im Mondlicht funkelte, entdeckte {childName} eine runde Steintür hinter silbernen Blättern.",
    },
    {
        "family": "hidden_dinosaur_valley",
        "en": "{childName} lived near a hidden valley where gentle dinosaurs slept among giant ferns and warm rain sparkled on broad green leaves.",
        "es": "En un valle escondido, donde dinosaurios tranquilos dormían entre helechos gigantes, {childName} vio una huella enorme llena de agua de lluvia brillante.",
        "fr": "Dans une vallée cachée où de doux dinosaures dormaient parmi des fougères géantes, {childName} vit une immense empreinte remplie d’eau de pluie lumineuse.",
        "it": "In una valle nascosta, dove dinosauri gentili dormivano tra felci giganti, {childName} vide un’enorme impronta piena di acqua piovana luminosa.",
        "de": "In einem versteckten Tal, in dem sanfte Dinosaurier zwischen riesigen Farnen schliefen, sah {childName} einen gewaltigen Fußabdruck voller leuchtendem Regenwasser.",
    },
    {
        "family": "ancient_observatory",
        "en": "{childName} lived near an ancient hilltop observatory, where a sleepy telescope watched the stars from a round stone tower.",
        "es": "En lo alto de un antiguo observatorio sobre una colina, {childName} vio cómo un telescopio adormecido se giraba solo hacia una estrella especialmente brillante.",
        "fr": "Au sommet d’un ancien observatoire perché sur une colline, {childName} vit un télescope endormi se tourner tout seul vers une étoile très brillante.",
        "it": "In cima a un antico osservatorio su una collina, {childName} vide un telescopio assonnato girarsi da solo verso una stella insolitamente luminosa.",
        "de": "Oben auf einer alten Sternwarte auf einem Hügel sah {childName}, wie sich ein schläfriges Teleskop von selbst zu einem ungewöhnlich hellen Stern drehte.",
    },
    {
        "family": "whale_island",
        "en": "{childName} lived on a tiny island shaped like a sleeping whale, where seashells lined the paths to a quiet lagoon.",
        "es": "En una pequeña isla con forma de ballena dormida, {childName} encontró conchas colocadas como una flecha que señalaba una laguna tranquila.",
        "fr": "Sur une petite île en forme de baleine endormie, {childName} trouva des coquillages disposés en flèche vers le lagon calme.",
        "it": "Su una piccola isola a forma di balena addormentata, {childName} trovò conchiglie disposte come una freccia verso la laguna tranquilla.",
        "de": "Auf einer kleinen Insel in Form eines schlafenden Wals fand {childName} Muscheln, die wie ein Pfeil zur stillen Lagune gelegt waren.",
    },
    {
        "family": "floating_cloud_city",
        "en": "{childName} lived in a quiet city floating above the clouds, where sky-ships drifted gently between shining towers.",
        "es": "En una ciudad tranquila que flotaba sobre las nubes, {childName} vio pequeñas ventanas encenderse como si despertaran mientras una campana suave sonaba por el cielo.",
        "fr": "Dans une ville calme flottant au-dessus des nuages, {childName} vit de petites fenêtres s’allumer comme si elles se réveillaient, tandis qu’une cloche douce sonnait dans le ciel.",
        "it": "In una città tranquilla che galleggiava sopra le nuvole, {childName} vide piccole finestre accendersi come se si svegliassero, mentre una campana dolce suonava nel cielo.",
        "de": "In einer stillen Stadt über den Wolken sah {childName}, wie kleine Fenster wach zu blinzeln begannen, während eine sanfte Glocke über den Himmel klang.",
    },
    {
        "family": "enchanted_hot_air_balloon",
        "en": "{childName} loved visiting a striped hot-air balloon that rested in a meadow of sleepy flowers, where the basket swayed gently in the breeze.",
        "es": "En un globo aerostático de rayas, posado en un prado de flores adormecidas, {childName} notó que la cesta tiraba suavemente hacia la luna.",
        "fr": "Dans une montgolfière rayée posée dans une prairie de fleurs endormies, {childName} sentit la nacelle tirer doucement vers la lune.",
        "it": "In una mongolfiera a righe appoggiata in un prato di fiori addormentati, {childName} sentì il cesto tirare piano verso la luna.",
        "de": "In einem gestreiften Heißluftballon, der auf einer Wiese schläfriger Blumen ruhte, spürte {childName}, wie der Korb sanft zum Mond zog.",
    },
    {
        "family": "glass_slipper_cafe",
        "en": "{childName} loved visiting the Glass Slipper Café, where warm cinnamon cakes cooled beside sparkling cups and the chairs politely tucked themselves in.",
        "es": "A {childName} le encantaba visitar el Café del Zapato de Cristal, donde los pasteles de canela se enfriaban junto a tazas brillantes y las sillas se colocaban solas con cuidado.",
        "fr": "{childName} adorait aller au Café du Soulier de Verre, où des gâteaux à la cannelle tiédaient près de tasses scintillantes et où les chaises se rangeaient toutes seules.",
        "it": "A {childName} piaceva visitare il Caffè della Scarpetta di Cristallo, dove dolci alla cannella si raffreddavano accanto a tazze scintillanti e le sedie si sistemavano da sole.",
        "de": "{childName} besuchte gern das Glasschuh-Café, wo warme Zimtküchlein neben funkelnden Tassen abkühlten und die Stühle sich höflich selbst zurechtrückten.",
    },
    {
        "family": "dragon_post_office",
        "en": "{childName} lived beside a tiny mountain post office where gentle dragons delivered moonlit letters tied with golden ribbon.",
        "es": "{childName} vivía junto a una pequeña oficina de correos en la montaña, donde unos dragones tranquilos repartían cartas iluminadas por la luna y atadas con cintas doradas.",
        "fr": "{childName} vivait près d’un minuscule bureau de poste de montagne, où de doux dragons livraient des lettres de lune nouées de rubans dorés.",
        "it": "{childName} viveva accanto a un piccolo ufficio postale di montagna, dove draghi gentili consegnavano lettere illuminate dalla luna e legate con nastri dorati.",
        "de": "{childName} wohnte neben einem winzigen Bergpostamt, wo sanfte Drachen mondhelle Briefe mit goldenen Bändern austrugen.",
    },
    {
        "family": "moon_bakery",
        "en": "{childName} visited a little moon bakery where crescent biscuits rested on silver trays and the ovens glowed like sleepy stars.",
        "es": "{childName} visitó una pequeña panadería de la luna, donde galletas con forma de media luna descansaban en bandejas plateadas y los hornos brillaban como estrellas adormecidas.",
        "fr": "{childName} entra dans une petite boulangerie de lune, où des biscuits en croissant reposaient sur des plateaux d’argent et où les fours luisaient comme des étoiles endormies.",
        "it": "{childName} visitò una piccola panetteria della luna, dove biscotti a mezzaluna riposavano su vassoi d’argento e i forni brillavano come stelle assonnate.",
        "de": "{childName} besuchte eine kleine Mondbäckerei, in der Mondsichelkekse auf silbernen Blechen lagen und die Öfen wie schläfrige Sterne leuchteten.",
    },
    {
        "family": "forest_school",
        "en": "{childName} went to a tiny woodland school where squirrels rang the bell with acorns and owls taught lessons beneath the quiet trees.",
        "es": "{childName} fue a una pequeña escuela del bosque, donde las ardillas tocaban la campana con bellotas y los búhos daban clase bajo los árboles tranquilos.",
        "fr": "{childName} allait dans une petite école de la forêt, où les écureuils sonnaient la cloche avec des glands et où les hiboux faisaient classe sous les arbres calmes.",
        "it": "{childName} andava in una piccola scuola del bosco, dove gli scoiattoli suonavano la campanella con le ghiande e i gufi facevano lezione sotto gli alberi tranquilli.",
        "de": "{childName} ging in eine kleine Waldschule, wo Eichhörnchen mit Eicheln die Glocke läuteten und Eulen unter stillen Bäumen unterrichteten.",
    },
    {
        "family": "toymaker_workshop",
        "en": "{childName} lived above a little toymaker’s workshop where wooden animals smiled whenever the lanterns were lit.",
        "es": "{childName} vivía sobre un pequeño taller de juguetes, donde los animales de madera sonreían cada vez que se encendían los farolillos.",
        "fr": "{childName} habitait au-dessus d’un petit atelier de jouets, où les animaux de bois souriaient dès que les lanternes s’allumaient.",
        "it": "{childName} viveva sopra una piccola bottega di giocattoli, dove gli animali di legno sorridevano ogni volta che si accendevano le lanterne.",
        "de": "{childName} wohnte über einer kleinen Spielzeugmacherwerkstatt, in der Holztiere lächelten, sobald die Laternen angezündet wurden.",
    },
    {
        "family": "star_painter_cottage",
        "en": "{childName} lived beside a small cottage where a gentle star painter coloured the evening sky with a silver brush.",
        "es": "{childName} vivía junto a una pequeña casita donde una pintora de estrellas coloreaba el cielo de la tarde con un pincel plateado.",
        "fr": "{childName} vivait près d’une petite maison où une douce peintre d’étoiles colorait le ciel du soir avec un pinceau d’argent.",
        "it": "{childName} viveva accanto a una casetta dove una gentile pittrice di stelle colorava il cielo della sera con un pennello d’argento.",
        "de": "{childName} wohnte neben einem kleinen Häuschen, in dem eine freundliche Sternenmalerin den Abendhimmel mit einem silbernen Pinsel färbte.",
    },
    {
        "family": "cloud_circus",
        "en": "{childName} saw a quiet cloud circus drifting above the rooftops, where soft drums whispered and acrobats balanced on moonbeams.",
        "es": "{childName} vio un circo tranquilo de nubes flotando sobre los tejados, donde unos tambores suaves susurraban y los acróbatas se equilibraban sobre rayos de luna.",
        "fr": "{childName} aperçut un cirque de nuages tout calme au-dessus des toits, où de doux tambours murmuraient et où les acrobates tenaient en équilibre sur des rayons de lune.",
        "it": "{childName} vide un tranquillo circo di nuvole sopra i tetti, dove tamburi leggeri sussurravano e gli acrobati camminavano sui raggi di luna.",
        "de": "{childName} sah einen stillen Wolkenzirkus über den Dächern treiben, wo leise Trommeln flüsterten und Akrobaten auf Mondstrahlen balancierten.",
    },
    {
        "family": "mermaid_library",
        "en": "{childName} discovered a hidden mermaid library beneath the calm waves, where shell books opened with a soft pearly glow.",
        "es": "{childName} descubrió una biblioteca de sirenas escondida bajo las olas tranquilas, donde los libros de concha se abrían con un brillo nacarado.",
        "fr": "{childName} découvrit une bibliothèque de sirènes cachée sous les vagues calmes, où les livres de coquillage s’ouvraient dans une douce lueur nacrée.",
        "it": "{childName} scoprì una biblioteca di sirene nascosta sotto le onde calme, dove libri di conchiglia si aprivano con un dolce bagliore di perla.",
        "de": "{childName} entdeckte unter den ruhigen Wellen eine versteckte Meerjungfrauen-Bibliothek, in der Muschelbücher in sanftem Perlglanz aufgingen.",
    },
    {
        "family": "rainbow_garden",
        "en": "{childName} found a rainbow garden behind a little gate, where every colour had its own gentle scent and the flowers hummed softly.",
        "es": "{childName} encontró un jardín arcoíris detrás de una pequeña verja, donde cada color tenía su propio aroma suave y las flores tarareaban despacio.",
        "fr": "{childName} trouva un jardin arc-en-ciel derrière un petit portail, où chaque couleur avait son doux parfum et où les fleurs fredonnaient tout bas.",
        "it": "{childName} trovò un giardino arcobaleno dietro un piccolo cancello, dove ogni colore aveva il suo profumo delicato e i fiori canticchiavano piano.",
        "de": "{childName} fand hinter einem kleinen Tor einen Regenbogengarten, in dem jede Farbe ihren eigenen sanften Duft hatte und die Blumen leise summten.",
    },
    {
        "family": "clockwork_greenhouse",
        "en": "{childName} visited a clockwork greenhouse where tiny brass watering cans marched carefully between sleepy moonflowers.",
        "es": "{childName} visitó un invernadero de relojería, donde pequeñas regaderas de latón caminaban con cuidado entre flores de luna adormecidas.",
        "fr": "{childName} visita une serre mécanique, où de petits arrosoirs de cuivre marchaient prudemment entre des fleurs de lune endormies.",
        "it": "{childName} visitò una serra a ingranaggi, dove piccoli annaffiatoi d’ottone camminavano con attenzione tra fiori di luna addormentati.",
        "de": "{childName} besuchte ein Uhrwerk-Gewächshaus, in dem winzige Messinggießkannen vorsichtig zwischen schläfrigen Mondblumen marschierten.",
    },
    {
        "family": "sleepy_puppet_theatre",
        "en": "{childName} helped at a tiny puppet theatre where velvet curtains opened only when the stars were ready to listen.",
        "es": "{childName} ayudaba en un pequeño teatro de marionetas, donde las cortinas de terciopelo solo se abrían cuando las estrellas estaban listas para escuchar.",
        "fr": "{childName} aidait dans un petit théâtre de marionnettes, où les rideaux de velours ne s’ouvraient que lorsque les étoiles étaient prêtes à écouter.",
        "it": "{childName} aiutava in un piccolo teatro di burattini, dove le tende di velluto si aprivano solo quando le stelle erano pronte ad ascoltare.",
        "de": "{childName} half in einem kleinen Puppentheater, in dem sich die Samtvorhänge erst öffneten, wenn die Sterne zum Zuhören bereit waren.",
    },
    {
        "family": "honeybee_palace",
        "en": "{childName} lived near a golden honeybee palace where tiny guards polished petals and every hallway smelled of warm honey.",
        "es": "{childName} vivía cerca de un palacio dorado de abejas, donde pequeños guardianes limpiaban pétalos y todos los pasillos olían a miel caliente.",
        "fr": "{childName} vivait près d’un palais doré d’abeilles, où de petits gardes polissaient les pétales et où chaque couloir sentait le miel tiède.",
        "it": "{childName} viveva vicino a un palazzo dorato delle api, dove piccole guardie lucidavano petali e ogni corridoio profumava di miele caldo.",
        "de": "{childName} wohnte nahe einem goldenen Honigbienenpalast, wo winzige Wachen Blütenblätter polierten und jeder Flur nach warmem Honig duftete.",
    },
    {
        "family": "dream_train_carriage",
        "en": "{childName} stepped into a quiet dream-train carriage where blue cushions lined the seats and the windows showed tomorrow’s sunrise.",
        "es": "{childName} entró en un tranquilo vagón del tren de los sueños, con cojines azules en los asientos y ventanas que mostraban el amanecer de mañana.",
        "fr": "{childName} monta dans un wagon calme du train des rêves, avec des coussins bleus sur les sièges et des fenêtres montrant le lever du soleil du lendemain.",
        "it": "{childName} salì su una tranquilla carrozza del treno dei sogni, con cuscini blu sui sedili e finestre che mostravano l’alba del giorno dopo.",
        "de": "{childName} stieg in einen stillen Traumzugwagen, in dem blaue Kissen auf den Sitzen lagen und die Fenster den Sonnenaufgang von morgen zeigten.",
    },
    {
        "family": "little_lighthouse_cafe",
        "en": "{childName} lived near a tiny lighthouse café where sailors drank cocoa and the lamp blinked kindly across the quiet sea.",
        "es": "{childName} vivía cerca de un pequeño café faro, donde los marineros tomaban cacao y la luz parpadeaba con cariño sobre el mar tranquilo.",
        "fr": "{childName} vivait près d’un minuscule café-phare, où les marins buvaient du chocolat chaud et où la lampe clignait doucement vers la mer calme.",
        "it": "{childName} viveva vicino a un piccolo caffè-faro, dove i marinai bevevano cacao e la lampada lampeggiava gentile sul mare tranquillo.",
        "de": "{childName} wohnte nahe einem winzigen Leuchtturm-Café, wo Seeleute Kakao tranken und die Lampe freundlich über das stille Meer blinkte.",
    },
    {
        "family": "dinosaur_kindergarten",
        "en": "{childName} visited a dinosaur kindergarten where baby dinosaurs practised tiny roars into soft moss pillows.",
        "es": "{childName} visitó una guardería de dinosaurios, donde los pequeños dinosaurios practicaban rugidos diminutos sobre cojines de musgo suave.",
        "fr": "{childName} visita une maternelle de dinosaures, où les bébés dinosaures s’exerçaient à pousser de tout petits rugissements dans des coussins de mousse.",
        "it": "{childName} visitò un asilo dei dinosauri, dove piccoli dinosauri provavano minuscoli ruggiti dentro cuscini di muschio morbido.",
        "de": "{childName} besuchte einen Dinosaurier-Kindergarten, in dem Babydinosaurier winzige Brüller in weiche Mooskissen übten.",
    },

]

# Backward-compatible English seed list kept for any older imports/tests.
OPENING_SEEDS = [seed["en"] for seed in OPENING_SEED_FAMILIES]


# Phase 10C: age-appropriate seed families.
# These rules keep the opening worlds matched to the child's comprehension level.
# Younger ages get familiar, low-load places; older ages may get broader fantasy worlds.
AGE_SEED_FAMILY_ALLOWLIST = {
    "0_2": {
        "pillow_harbour",
        "hidden_garden_gate",
        "meadow_clock",
        "moon_bakery",
        "dinosaur_kindergarten",
    },
    "3_4": {
        "pillow_harbour",
        "hidden_garden_gate",
        "moon_bakery",
        "forest_school",
        "toymaker_workshop",
        "dinosaur_kindergarten",
        "rainbow_garden",
        "sleepy_forest_path",
        "lantern_treehouse",
    },
    "5_6": {
        "sleepy_forest_path",
        "lantern_treehouse",
        "hidden_garden_gate",
        "moonlit_library",
        "seaside_cave",
        "sleepy_castle_hall",
        "glowing_attic",
        "river_of_stars",
        "pillow_harbour",
        "dragon_market",
        "glass_slipper_cafe",
        "dragon_post_office",
        "moon_bakery",
        "forest_school",
        "toymaker_workshop",
        "star_painter_cottage",
        "cloud_circus",
        "rainbow_garden",
        "sleepy_puppet_theatre",
        "honeybee_palace",
        "little_lighthouse_cafe",
        "dinosaur_kindergarten",
    },
    "7_8": {
        "sleepy_forest_path",
        "lantern_treehouse",
        "hidden_garden_gate",
        "cloud_island",
        "moonlit_library",
        "seaside_cave",
        "sleepy_castle_hall",
        "glowing_attic",
        "snowy_village",
        "river_of_stars",
        "meadow_clock",
        "pillow_harbour",
        "underwater_palace",
        "pirate_harbour",
        "dragon_market",
        "northern_lights_village",
        "sky_train_station",
        "whale_island",
        "enchanted_hot_air_balloon",
        "glass_slipper_cafe",
        "dragon_post_office",
        "moon_bakery",
        "forest_school",
        "toymaker_workshop",
        "star_painter_cottage",
        "cloud_circus",
        "mermaid_library",
        "rainbow_garden",
        "clockwork_greenhouse",
        "sleepy_puppet_theatre",
        "honeybee_palace",
        "dream_train_carriage",
        "little_lighthouse_cafe",
        "dinosaur_kindergarten",
    },
    "9_10": {
        "sleepy_forest_path",
        "lantern_treehouse",
        "hidden_garden_gate",
        "cloud_island",
        "moonlit_library",
        "seaside_cave",
        "sleepy_castle_hall",
        "glowing_attic",
        "snowy_village",
        "river_of_stars",
        "meadow_clock",
        "amazon_treehouse",
        "nile_river_boat",
        "desert_caravan",
        "underwater_palace",
        "pirate_harbour",
        "dragon_market",
        "crystal_cavern",
        "northern_lights_village",
        "sky_train_station",
        "jungle_waterfall",
        "hidden_dinosaur_valley",
        "whale_island",
        "floating_cloud_city",
        "enchanted_hot_air_balloon",
        "glass_slipper_cafe",
        "dragon_post_office",
        "moon_bakery",
        "forest_school",
        "toymaker_workshop",
        "star_painter_cottage",
        "cloud_circus",
        "mermaid_library",
        "rainbow_garden",
        "clockwork_greenhouse",
        "sleepy_puppet_theatre",
        "honeybee_palace",
        "dream_train_carriage",
        "little_lighthouse_cafe",
    },
    "11_12": set(),
}

AGE_HUMOUR_PROFILES = {
    "0_2": {
        "quirks": [
            "likes saying hello to animals",
            "claps when something wobbles",
            "keeps a soft toy close",
            "giggles at funny sounds",
        ],
        "events": [
            "a duck says the wrong sound and everyone smiles",
            "a cushion gives a tiny squeak and points the way",
            "a sleepy animal wears a blanket like a hat",
        ],
        "instruction": "Use one very simple smile moment: a funny sound, a wobbly hat, or an animal doing one silly thing.",
    },
    "3_4": {
        "quirks": [
            "loves funny animal sounds",
            "counts snacks out loud",
            "waves at every animal",
            "keeps a favourite pebble in a pocket",
        ],
        "events": [
            "a small animal puts a hat on backwards and walks the wrong way",
            "a dragon sneezes one bubble that lands on someone's nose",
            "a rabbit takes an instruction too literally and carries a spoon like a flag",
        ],
        "instruction": "Use 1-2 clear visual giggles that a preschool child can picture immediately.",
    },
    "5_6": {
        "quirks": [
            "packs one unnecessary snack for emergencies",
            "names every tiny creature they meet",
            "talks to important objects before using them",
            "counts steps when thinking",
        ],
        "events": [
            "a helper misunderstands the plan and proudly brings the wrong object first",
            "a dragon sneezes a harmless puff that reveals the next clue",
            "a squirrel's oversized hat slips down and accidentally shows where to look",
            "a map folds itself into a paper bird and lands on the useful clue",
        ],
        "instruction": "Use two warm giggle moments: one visual mishap and one simple misunderstanding. At least one must help the plot.",
    },
    "7_8": {
        "quirks": [
            "keeps tiny notes in unexpected pockets",
            "cannot resist making things neat before solving them",
            "collects unusual buttons",
            "gives serious names to silly objects",
        ],
        "events": [
            "a side character follows the instruction exactly but in the least helpful way",
            "a clue is discovered because someone trips over their own overpacked bag",
            "a grumpy helper says a dramatic phrase at completely the wrong time",
            "a magical object behaves like a stubborn pet before revealing its purpose",
        ],
        "instruction": "Use character-based humour that still feels gentle; the joke should reveal personality or create a useful clue.",
    },
    "9_10": {
        "quirks": [
            "keeps a notebook of odd rules",
            "notices patterns other people miss",
            "uses overly careful plans that need changing",
            "gives practical advice to very impractical creatures",
        ],
        "events": [
            "a formal rule is misunderstood in a funny but useful way",
            "a character's over-complicated plan fails, then inspires a simpler solution",
            "a serious helper is embarrassed by a harmless but revealing mistake",
            "a strange tradition creates a comic obstacle that the child must interpret",
        ],
        "instruction": "Use smarter situational humour, but keep it warm, not sarcastic. The humour should expose the real problem.",
    },
    "11_12": {
        "quirks": [
            "notices contradictions in rules",
            "uses dry but kind observations",
            "keeps calm when adults overcomplicate things",
            "tests ideas before trusting them",
        ],
        "events": [
            "a grand tradition turns out to be based on a small misunderstanding",
            "an impressive title hides a very ordinary problem",
            "a complicated system fails because everyone ignored a simple detail",
            "a formal ceremony goes slightly wrong in a way that reveals the solution",
        ],
        "instruction": "Use age-older wit through irony, over-formality, or clever misunderstanding, but avoid sarcasm or meanness.",
    },
}

# Phase 7: weighted away from object-retrieval as the default.
# Object stories still exist, but emotional / relationship / mystery shapes should win more often.
STORY_ARCHETYPE_WEIGHTS = [
    ("mystery_to_solve", 15),
    ("rescue_mission", 13),
    ("secret_hidden_place", 11),
    ("helping_rivals_become_friends", 11),
    ("preparing_for_celebration", 10),
    ("mistaken_identity", 9),
    ("two_paths_choice", 9),
    ("delivery_mission", 8),
    ("race_against_time", 7),
    ("magical_competition", 6),
    ("lost_object", 4),
    ("broken_spell_or_magic", 4),
    ("treasure_hunt", 3),
    ("repair_something_magical", 3),
]

EMOTIONAL_STORY_TYPES = [
    "friendship",
    "courage",
    "kindness",
    "celebration",
    "mystery",
    "helping someone",
    "family adventure",
    "making a mistake and putting it right",
    "learning patience",
    "discovering a hidden talent",
    "funny misunderstanding",
    "teamwork",
    "welcoming someone new",
    "sharing something precious",
    "keeping a promise",
    "asking for help",
]

CHARACTER_TRAITS = [
    "shy",
    "brave",
    "clumsy",
    "curious",
    "impatient",
    "funny",
    "forgetful",
    "gentle",
    "confident",
    "creative",
    "cheerful",
    "nervous",
    "determined",
    "dreamy",
]

FUNNY_QUIRKS = [
    "loves sandwiches",
    "collects shiny buttons",
    "always wears boots",
    "sings while working",
    "is scared of butterflies",
    "carries too many snacks",
    "forgets names",
    "speaks in rhymes",
    "cannot stop collecting pebbles",
    "thinks every cloud looks like a sandwich",
    "always packs far too many snacks",
    "names every ladybird they meet",
    "believes hats have feelings",
    "keeps talking to their shoes",
    "cannot resist counting things",
    "waves at every fish",
    "tries to teach birds to sing",
    "puts important things in strange places",
]

COMFORT_HABITS = [
    "twists a sleeve when thinking",
    "keeps a lucky pebble in a pocket",
    "counts steps when nervous",
    "hums softly while solving problems",
    "taps fingers together when excited",
    "smooths the corner of a blanket",
    "keeps tiny treasures in a pouch",
    "whispers ideas to the stars",
]

SIGNATURE_BEHAVIOURS = [
    "always checks their hat before speaking",
    "collects unusual buttons",
    "writes everything in a notebook",
    "carries far too many snacks",
    "cannot resist solving riddles",
    "polishes spectacles that are never dirty",
    "talks to flowers as if they can answer",
    "keeps pockets full of interesting things",
    "alphabetises biscuits before eating them",
    "wears three scarves even indoors",
    "is scared of seagulls but pretends not to be",
    "keeps forgetting where they put their hat",
    "paints vegetables instead of pictures",
]

FAVOURITE_PHRASES = [
    "Stars and teacups!",
    "Well butter my biscuits!",
    "Good feathers!",
    "Oh my moonbeams!",
    "What a curious thing!",
    "By the sleepy sea!",
    "Biscuit crumbs and dragon tails!",
]

PLOT_HUMOUR_EVENTS = [
    "a dragon sneezes and accidentally reveals a secret door",
    "a rabbit loses its spoon and discovers an important clue",
    "a squirrel wears the wrong hat and causes a funny misunderstanding",
    "a teacup rolls away and leads everyone to a hidden place",
    "a map gets folded into a paper bird and flies off",
    "a shy owl gives the wrong directions at first",
    "a bear mistakes a recipe for a treasure map",
    "a tiny boat floats away carrying an important message",
    "a bell tree rings because someone sat on a branch",
    "a sleepy badger fixes the wrong thing before finding the real problem",
]

STORY_ARCHETYPE_INSTRUCTIONS = {
    "mystery_to_solve": {
        "label": "Mystery to Solve",
        "rules": [
            "Begin with a question, clue, strange change, missing sign, unusual sound, or confusing event.",
            "Include 2-3 gentle clues before the answer becomes clear.",
            "The child should notice details and connect clues rather than simply being told the answer.",
            "The resolution should reveal why the mystery happened and include a warm, reassuring outcome.",
        ],
    },
    "rescue_mission": {
        "label": "Rescue Mission",
        "rules": [
            "Someone or something needs safe, gentle help, but there must be no frightening danger.",
            "The first rescue idea should not fully work or should reveal a second step.",
            "The child should use kindness, patience, creativity, or teamwork to solve the problem.",
            "End with relief, gratitude shown through action, and a concrete callback image.",
        ],
    },
    "lost_object": {
        "label": "Lost Object",
        "rules": [
            "An important object is missing and the world cannot work properly without it.",
            "Follow clues, wrong turns, or mistaken assumptions before finding the truth.",
            "The object should matter emotionally or practically, not just be treasure.",
            "The object should return in the final image or ending callback.",
        ],
    },
    "secret_hidden_place": {
        "label": "Secret Hidden Place",
        "rules": [
            "The story leads to a place that is hidden, forgotten, or only opens in a special way.",
            "The hidden place must have its own simple rule, tradition, or problem.",
            "The child should earn entry through a choice, kindness, observation, or courage.",
            "The ending should suggest the place still exists after bedtime.",
        ],
    },
    "broken_spell_or_magic": {
        "label": "Broken Spell or Broken Magic",
        "rules": [
            "Something magical is not working correctly: a spell, bridge, map, fountain, lantern, song, or sky-sign.",
            "The first fix should not fully work and should reveal what is really wrong.",
            "The solution should require the moral in action rather than a magic word alone.",
            "Show the magic working again through a specific visual change.",
        ],
    },
    "delivery_mission": {
        "label": "Delivery Mission",
        "rules": [
            "The child must deliver something meaningful before it is needed.",
            "The journey should include one obstacle, delay, or choice about helping someone else on the way.",
            "The delivered item should solve a concrete bedtime-safe problem.",
            "End with the delivered item creating a final comforting image.",
        ],
    },
    "treasure_hunt": {
        "label": "Treasure Hunt",
        "rules": [
            "The treasure should be surprising, meaningful, or useful, not just gold or jewels.",
            "Use clues, maps, riddles, marks, or signs that the child can follow.",
            "Include one misleading clue or obstacle before the real treasure is found.",
            "The treasure should connect to the moral and final callback.",
        ],
    },
    "two_paths_choice": {
        "label": "Two Paths / Difficult Choice",
        "rules": [
            "At some point the child must choose between two reasonable options.",
            "Neither choice should be obviously perfect; each should have a small cost or consequence.",
            "The story should show why the chosen path matters through events, not explanation.",
            "The ending should gently honour the choice the child made.",
        ],
    },
    "repair_something_magical": {
        "label": "Repair Something Magical",
        "rules": [
            "A magical object, place, machine, instrument, garden, bridge, or vehicle needs repairing.",
            "The child should gather or discover the missing piece through action.",
            "A first repair attempt should partly work, then reveal what still needs attention.",
            "The completed repair should create a memorable final image.",
        ],
    },
    "helping_rivals_become_friends": {
        "label": "Helping Rivals Become Friends",
        "rules": [
            "Two characters want different things or misunderstand one another.",
            "The child should listen, notice what each side needs, and help them cooperate.",
            "Avoid lectures; show the friendship through shared action.",
            "End with the two characters doing something together that they could not do alone.",
        ],
    },
    "race_against_time": {
        "label": "Race Against Time",
        "rules": [
            "There is a soft bedtime-safe countdown: before moonrise, before the last lantern dims, before the tide turns, or before the celebration begins.",
            "Keep urgency gentle, not frightening.",
            "Include one delay that forces the child to choose what matters most.",
            "The resolution should arrive just in time and then slow into a calm ending.",
        ],
    },
    "magical_competition": {
        "label": "Magical Competition",
        "rules": [
            "The story includes a friendly contest, performance, game, or challenge with clear rounds or tasks.",
            "Winning should not be the main lesson; kindness, creativity, honesty, or teamwork should matter more.",
            "Include one setback during the competition.",
            "End with a shared celebration or unexpected prize connected to the adventure.",
        ],
    },
    "mistaken_identity": {
        "label": "Mistaken Identity",
        "rules": [
            "Someone or something is mistaken for the wrong person, creature, object, or sign.",
            "The confusion should create funny or curious complications, not fear.",
            "The child should uncover the truth by paying attention and asking kindly.",
            "The reveal should help characters understand each other better.",
        ],
    },
    "preparing_for_celebration": {
        "label": "Preparing for a Celebration",
        "rules": [
            "A magical celebration, show, feast, parade, or ceremony needs help getting ready.",
            "Something important should go wrong before the celebration can begin.",
            "The child should solve the problem through the chosen moral in action.",
            "End with one vivid celebration image that feels cosy and earned.",
        ],
    },
}

FIRST_PAGE_TIMEOUT_SECONDS = 30
# User-facing consistency target: if Gemini has not produced page 1
# quickly enough, return a deterministic page-1 fallback so Reader can open.
# The full story still completes in the normal background Gemini path.
FIRST_PAGE_SOFT_LIMIT_SECONDS = 22

# Background continuation is generated in small batches so pages 2+ can
# become available to the reader sooner. This preserves Page-1-first playback
# while avoiding one long Gemini call blocking all remaining pages.
BACKGROUND_PAGE_BATCH_SIZE = 3
# If Gemini truncates a continuation batch, reduce the request size instead of
# leaving the story stuck on a partial page count. This only affects background
# pages 2+ and never changes Page-1-first narration behaviour.
BACKGROUND_RECOVERY_BATCH_SIZES = [2, 1]

# Keep Page 1 generation small and fast. This only affects the initial
# Gemini Page 1 call; background continuation keeps its normal generation.
# 768 gives enough headroom for valid JSON wrapper + title + 500-650 chars
# while still keeping Page 1 small for the Page-1-first performance rule.
FIRST_PAGE_MAX_OUTPUT_TOKENS = 768

# Per-call Gemini configs. Keep these local and explicit because PillowTales
# uses Gemini for different jobs: Page 1 speed, continuation creativity, full
# fallback generation, and deterministic metadata extraction. Do not configure
# the model globally with one setting for every task.
FIRST_PAGE_GENERATION_CONFIG = {
    "temperature": 0.85,
    "top_p": 0.95,
    "max_output_tokens": FIRST_PAGE_MAX_OUTPUT_TOKENS,
    "response_mime_type": "application/json",
}

REMAINING_PAGES_GENERATION_CONFIG = {
    "temperature": 0.95,
    "top_p": 0.95,
    "max_output_tokens": 4096,
    "response_mime_type": "application/json",
}

FULL_STORY_GENERATION_CONFIG = {
    "temperature": 0.95,
    "top_p": 0.95,
    "max_output_tokens": 8192,
    "response_mime_type": "application/json",
}

METADATA_GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": 0.8,
    "max_output_tokens": 768,
    "response_mime_type": "application/json",
}


class StoryService:
    def __init__(self, story_repo: StoryRepository):
        self.story_repo = story_repo
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model) if settings.gemini_api_key else None

    def _log_gemini_response_metadata(self, label: str, response: Any) -> None:
        """Log lightweight Gemini metadata without exposing full story text.

        This is diagnostic only. It helps identify MAX_TOKENS, SAFETY, empty
        candidates, or SDK response issues when JSON parsing fails. It does not
        change prompts, narration, chunking, polling, storage, or reader flow.
        """
        try:
            candidates = getattr(response, 'candidates', None) or []
            candidate_count = len(candidates) if hasattr(candidates, '__len__') else 0
            prompt_feedback = getattr(response, 'prompt_feedback', None)
            usage_metadata = getattr(response, 'usage_metadata', None)
            text_value = getattr(response, 'text', '') or ''
            print(
                f"[DEBUG] Gemini metadata label={label} "
                f"candidates={candidate_count} response_chars={len(text_value)} "
                f"prompt_feedback={prompt_feedback} usage={usage_metadata}"
            )

            for idx, candidate in enumerate(candidates[:2]):
                finish_reason = getattr(candidate, 'finish_reason', None)
                safety_ratings = getattr(candidate, 'safety_ratings', None)
                print(
                    f"[DEBUG] Gemini candidate label={label} index={idx} "
                    f"finish_reason={finish_reason} safety_ratings={safety_ratings}"
                )
        except Exception as metadata_exc:
            print(f"[DEBUG] Gemini metadata log skipped label={label}: {metadata_exc}")

    def _select_companion(self, request: GenerateStoryRequest, subscription: SubscriptionResponse) -> Optional[dict]:
        # V1 production focus: do not randomly introduce companions.
        # The frontend currently uses the PillowTales bear as the single visual anchor.
        # We still honour an explicit valid companionId from existing clients/backward compatibility,
        # but no longer auto-select a random companion when none is requested.
        if request.companionId and request.companionId in STORY_COMPANIONS:
            companion = STORY_COMPANIONS[request.companionId].copy()
            companion['id'] = request.companionId
            return companion
        return None

    def _localized_dict_value(self, value: Any, language_code: str, fallback: str = "") -> str:
        if isinstance(value, dict):
            lang = (language_code or "en").lower()[:2]
            return str(value.get(lang) or value.get("en") or fallback or "")
        return str(value or fallback or "")

    def _localized_companion(self, companion: Optional[dict], language_code: Optional[str]) -> Optional[dict]:
        if not companion:
            return None

        lang = (language_code or "en").lower()[:2]
        localized = companion.copy()
        localized["name"] = self._localized_dict_value(
            companion.get("name_i18n"),
            lang,
            companion.get("name", ""),
        )
        localized["description"] = self._localized_dict_value(
            companion.get("description_i18n"),
            lang,
            companion.get("description", ""),
        )
        return localized

    def _localized_theme_label(self, theme: Optional[str], language_code: Optional[str]) -> str:
        raw = str(theme or "").strip()
        if not raw:
            return raw

        lang = (language_code or "en").lower()[:2]
        key = raw.lower().replace("-", "_").replace(" ", "_")
        theme_labels = {
            "dragons": {"en": "dragons", "es": "dragones", "fr": "dragons", "de": "Drachen", "it": "draghi"},
            "space": {"en": "space", "es": "espacio", "fr": "espace", "de": "Weltraum", "it": "spazio"},
            "animals": {"en": "animals", "es": "animales", "fr": "animaux", "de": "Tiere", "it": "animali"},
            "princess": {"en": "princess", "es": "princesa", "fr": "princesse", "de": "Prinzessin", "it": "principessa"},
            "adventure": {"en": "adventure", "es": "aventura", "fr": "aventure", "de": "Abenteuer", "it": "avventura"},
            "underwater": {"en": "underwater", "es": "bajo el agua", "fr": "sous l’eau", "de": "Unterwasserwelt", "it": "mondo sottomarino"},
            "forest": {"en": "forest", "es": "bosque", "fr": "forêt", "de": "Wald", "it": "foresta"},
            "magic": {"en": "magic", "es": "magia", "fr": "magie", "de": "Magie", "it": "magia"},
            "dinosaurs": {"en": "dinosaurs", "es": "dinosaurios", "fr": "dinosaures", "de": "Dinosaurier", "it": "dinosauri"},
            "superheroes": {"en": "superheroes", "es": "superhéroes", "fr": "super-héros", "de": "Superhelden", "it": "supereroi"},
            "emotions": {"en": "emotions", "es": "emociones", "fr": "émotions", "de": "Gefühle", "it": "emozioni"},
        }
        return theme_labels.get(key, {}).get(lang) or theme_labels.get(key, {}).get("en") or raw

    def _localized_relationship_label(self, relationship: Optional[str], language_code: Optional[str]) -> str:
        raw = str(relationship or "").strip()
        if not raw:
            return raw

        key = raw.lower().replace("-", "_").replace(" ", "_")
        lang = (language_code or "en").lower()[:2]
        relationship_labels = {
            "mother": {"en": "mother", "es": "madre", "fr": "mère", "de": "Mutter", "it": "mamma"},
            "mum": {"en": "mum", "es": "mamá", "fr": "maman", "de": "Mama", "it": "mamma"},
            "mom": {"en": "mum", "es": "mamá", "fr": "maman", "de": "Mama", "it": "mamma"},
            "father": {"en": "father", "es": "padre", "fr": "père", "de": "Vater", "it": "papà"},
            "dad": {"en": "dad", "es": "papá", "fr": "papa", "de": "Papa", "it": "papà"},
            "sister": {"en": "sister", "es": "hermana", "fr": "sœur", "de": "Schwester", "it": "sorella"},
            "brother": {"en": "brother", "es": "hermano", "fr": "frère", "de": "Bruder", "it": "fratello"},
            "friend": {"en": "friend", "es": "amigo o amiga", "fr": "ami ou amie", "de": "Freund oder Freundin", "it": "amico o amica"},
            "cat": {"en": "cat", "es": "gato", "fr": "chat", "de": "Katze", "it": "gatto"},
            "dog": {"en": "dog", "es": "perro", "fr": "chien", "de": "Hund", "it": "cane"},
            "pet": {"en": "pet", "es": "mascota", "fr": "animal de compagnie", "de": "Haustier", "it": "animale domestico"},
        }
        return relationship_labels.get(key, {}).get(lang) or relationship_labels.get(key, {}).get("en") or raw

    def _no_companion_required_text(self, language_code: Optional[str]) -> str:
        return {
            "en": "No companion is required.",
            "es": "No hace falta incluir ningún compañero.",
            "fr": "Aucun compagnon n’est nécessaire.",
            "de": "Es muss kein Begleiter vorkommen.",
            "it": "Non è necessario includere un compagno.",
        }.get((language_code or "en").lower()[:2], "No companion is required.")

    def _no_extra_characters_required_text(self, language_code: Optional[str]) -> str:
        return {
            "en": "No extra family members or friends are required.",
            "es": "No hace falta incluir familiares, amistades ni mascotas adicionales.",
            "fr": "Aucun membre de la famille, ami ou animal supplémentaire n’est nécessaire.",
            "de": "Es müssen keine zusätzlichen Familienmitglieder, Freunde oder Haustiere vorkommen.",
            "it": "Non è necessario includere altri familiari, amici o animali.",
        }.get((language_code or "en").lower()[:2], "No extra family members or friends are required.")

    def _select_story_archetype(self) -> dict:
        """Choose a hidden weighted story archetype for plot variety.

        This is local, instant, and does not touch narration, chunking, or
        reader flow. The archetype is prompt guidance only.
        """
        keys = [item[0] for item in STORY_ARCHETYPE_WEIGHTS]
        weights = [item[1] for item in STORY_ARCHETYPE_WEIGHTS]
        selected_key = random.choices(keys, weights=weights, k=1)[0]
        return STORY_ARCHETYPE_INSTRUCTIONS[selected_key]

    def _story_archetype_block(self, archetype: Optional[dict]) -> str:
        if not archetype:
            return ""

        rules = archetype.get("rules") or []
        rule_lines = "\n".join(f"- {rule}" for rule in rules)
        label = archetype.get("label", "Story Archetype")
        return f"""SELECTED STORY ARCHETYPE:
- This story MUST primarily follow the "{label}" archetype.
- Use this archetype to shape the opening promise, middle obstacle, resolution, and final callback.
- Do not drift into the default pattern of finding a creature, helping it once, receiving a gift, and going home unless that is genuinely required by this archetype.
{rule_lines}
"""

    def _select_emotional_story_type(self) -> str:
        """Choose a hidden emotional driver for story diversity.

        Local only: no network call and no narration/chunking impact.
        """
        return random.choice(EMOTIONAL_STORY_TYPES)

    def _emotional_story_block(self, emotional_theme: Optional[str]) -> str:
        if not emotional_theme:
            return ""
        return f"""SELECTED EMOTIONAL ENGINE:
- The central emotional theme of this story should be: {emotional_theme}.
- The emotional journey should be more important than any magical object.
- If a magical item exists, it should support the relationship, choice, courage, kindness, or problem-solving arc rather than become the whole goal.
- The ending should primarily resolve an emotional need, promise, worry, friendship, misunderstanding, or act of courage rather than simply returning or fixing an object.
"""

    def _select_character_trait(self) -> str:
        """Choose a hidden character trait for story personality variety."""
        return random.choice(CHARACTER_TRAITS)

    def _select_funny_quirk(self) -> str:
        """Choose a hidden comic quirk for gentle bedtime humour variety."""
        return random.choice(FUNNY_QUIRKS)
    def _select_comfort_habit(self) -> str:
        return random.choice(COMFORT_HABITS)

    def _select_signature_behaviour(self) -> str:
        return random.choice(SIGNATURE_BEHAVIOURS)

    def _select_favourite_phrase(self) -> str:
        return random.choice(FAVOURITE_PHRASES)

    def _select_plot_humour_event(self) -> str:
        return random.choice(PLOT_HUMOUR_EVENTS)

    def _personality_humour_block(
        self,
        character_trait: Optional[str],
        funny_quirk: Optional[str],
        age: Any,
    ) -> str:
        comfort_habit = self._select_comfort_habit()
        signature = self._select_signature_behaviour()
        phrase = self._select_favourite_phrase()
        plot_humour = self._select_age_plot_humour_event(age)
        age_humour_instruction = self._age_humour_instruction(age)

        return f"""
SELECTED CHARACTER PERSONALITY ENGINE:
- Give the child or one important side character a clear personality trait: {character_trait}.
- Give the child a memorable quirk: {funny_quirk}.
- Give the child a comfort habit: {comfort_habit}.
- At least one of these should influence the story solution.

SIDE CHARACTER RULES:
- One important side character should have this distinctive behaviour:
  {signature}

- That character may occasionally say:
  "{phrase}"

PLOT HUMOUR RULES:
- Age-specific humour guidance: {age_humour_instruction}
- Include this funny event somewhere in the story:
  {plot_humour}

- The funny event MUST change what happens next.
- The funny event should create a clue, obstacle, solution, or new discovery.
- Do not include humour that can be removed without affecting the story.
- At least one supporting character should have a recurring funny behaviour that appears more than once.

SHOW DON'T TELL RULES:
- Never describe a character only as kind, brave, curious, or gentle.
- Show personality through actions, choices, dialogue, mistakes, and habits.
- Let quirks create gentle humour naturally.
- Include at least one funny misunderstanding or unexpected behaviour caused by a quirk.
- The humour should make a parent smile and may make a child giggle.
- The ending should feel earned because of the hero's personality.
- Avoid slapstick, sarcasm, teasing, or loud comedy.
"""



    def _safe_child_age(self, age: Any) -> int:
        """Return a bounded child age for prompt guidance only.

        This is deliberately local and prompt-only. It does not affect API
        contracts, page count, narration, polling, storage, or subscriptions.
        """
        try:
            parsed = int(age)
        except (TypeError, ValueError):
            parsed = 6
        return max(0, min(parsed, 12))

    def _oxford_inspired_age_profile_block(self, age: Any) -> str:
        """Internal age calibration inspired by Oxford Owl reading progression.

        This does not copy Oxford Reading Tree content or style. It is only a
        developmental guide for sentence length, vocabulary load, dialogue,
        plot complexity, emotional range, and humour. Prompt-only: no narration,
        chunking, polling, storage, subscriptions, or reader behaviour changes.
        """
        child_age = self._safe_child_age(age)

        if child_age <= 2:
            profile = """OXFORD-INSPIRED AGE PROFILE — AGE 0-2:
- Reading/listening stage: earliest shared read-aloud and nursery-rhythm level.
- Sentence shape: very short, one idea per sentence, mostly 3-8 words.
- Vocabulary: almost entirely familiar concrete words, sounds, colours, animals, body actions, bedtime objects, and family words.
- Dialogue: minimal; short phrases only.
- Plot: one place, one tiny event, one comfort action.
- Emotion: happy, sad, sleepy, surprised, cosy. Show through cuddles, looking, reaching, hiding, or sounds.
- Humour: one visual or sound-based smile moment.
- New words: almost none; any new word must be obvious from context.
"""
        elif child_age <= 4:
            profile = """OXFORD-INSPIRED AGE PROFILE — AGE 3-4:
- Reading/listening stage: early picture-book comprehension and simple patterned language.
- Sentence shape: short, clear sentences, usually 5-10 words, with occasional repetition.
- Vocabulary: familiar everyday words plus a few simple storybook words such as cosy, twinkle, whisper, surprise, or sparkle when concrete.
- Dialogue: short, direct lines that a young child can repeat.
- Plot: one clear place, one simple problem, one helper, one solution path.
- Emotion: happy, worried, scared, proud, kind, brave. Show through simple actions.
- Humour: obvious visual silliness, animal behaviour, wrong hats, funny sounds, or simple misunderstandings.
- New words: one or two only, supported by the sentence around them.
"""
        elif child_age <= 6:
            profile = """OXFORD-INSPIRED AGE PROFILE — AGE 5-6:
- Reading/listening stage: early independent-reader clarity with richer read-aloud adventure.
- Sentence shape: mostly 6-12 words, with occasional longer sentences when very clear.
- Vocabulary: familiar action words plus gentle story vocabulary such as curious, discovered, puzzled, patient, invitation, bridge, promise, clue.
- Dialogue: regular but simple; dialogue should reveal what someone needs, notices, or misunderstands.
- Plot: one main goal, one main helper, one obstacle, one first idea that may partly fail.
- Emotion: worried, shy, disappointed, brave, proud, patient. Show through choices and behaviour.
- Humour: visual mishaps and simple misunderstandings that affect the plot.
- New words: a small number per page, always clear from context.
"""
        elif child_age <= 8:
            profile = """OXFORD-INSPIRED AGE PROFILE — AGE 7-8:
- Reading/listening stage: confident early chapter-book feel while remaining bedtime clear.
- Sentence shape: varied but readable, usually 10-18 words.
- Vocabulary: richer but still child-friendly words such as pattern, narrow, secret, festival, invention, nervous, proud, practice, promise, clue.
- Dialogue: more frequent and characterful; characters may disagree gently, ask questions, or reveal motives.
- Plot: connected scenes, 2-3 clues or steps, a clear midpoint complication, and a child-led decision.
- Emotion: confused, jealous, nervous, determined, left out, responsible, relieved. Show through dialogue and choices.
- Humour: character habits, literal misunderstandings, over-serious helpers, or repeated funny behaviour with payoff.
- New words: welcome when useful, but action must stay easy to follow.
"""
        elif child_age <= 10:
            profile = """OXFORD-INSPIRED AGE PROFILE — AGE 9-10:
- Reading/listening stage: richer middle-grade-style bedtime story with controlled complexity.
- Sentence shape: varied sentences, often 12-22 words, but never dense or adult.
- Vocabulary: allow more precise words such as investigate, tradition, responsibility, generous, cautious, suspicious, determined, solution.
- Dialogue: should reveal motives, pressure, uncertainty, or changing trust.
- Plot: one main thread with a small subplot or deeper choice when useful.
- Emotion: loyalty, guilt, fairness, pressure, confidence, regret, responsibility. Keep it hopeful and bedtime-safe.
- Humour: smarter situational humour, over-complicated plans, rules misunderstood, or formal traditions going wrong.
- New words: acceptable if they support story richness and do not slow comprehension.
"""
        else:
            profile = """OXFORD-INSPIRED AGE PROFILE — AGE 11-12:
- Reading/listening stage: upper-child storytelling with nuance, but still warm bedtime fiction.
- Sentence shape: fluent and varied, with longer sentences allowed when natural and clear.
- Vocabulary: richer words such as uncertainty, consequence, reluctant, contradiction, evidence, independence, forgiveness, thoughtful.
- Dialogue: more layered; characters can imply feelings without explaining everything.
- Plot: nuanced motives, a stronger mystery or choice, and clear consequences, but no grim or teen-focused themes.
- Emotion: uncertainty, responsibility, independence, loyalty, forgiveness, self-doubt, confidence.
- Humour: gentle wit, irony of rules, over-formality, or clever misunderstanding, never sarcasm or meanness.
- New words: richer vocabulary is allowed, but the story must still read aloud smoothly.
"""

        return profile + """
GENERAL OXFORD-INSPIRED CALIBRATION RULE:
- Use these profiles only as developmental guidance for language complexity.
- Do not copy or imitate Oxford Reading Tree stories, characters, wording, plots, or branded style.
- The story must remain original PillowTales bedtime fiction.
- Age should change more than vocabulary: it should change sentence rhythm, dialogue, plot load, emotional depth, humour, and how much the child must infer.
"""

    def _age_band_key(self, age: Any) -> str:
        child_age = self._safe_child_age(age)
        if child_age <= 2:
            return "0_2"
        if child_age <= 4:
            return "3_4"
        if child_age <= 6:
            return "5_6"
        if child_age <= 8:
            return "7_8"
        if child_age <= 10:
            return "9_10"
        return "11_12"

    def _seed_pool_for_age(self, age: Any) -> list[dict]:
        """Return opening seeds suitable for the child's age.

        Age-appropriate seeds reduce cognitive load for younger children by
        avoiding complex, faraway, multi-concept settings until older ages.
        """
        band = self._age_band_key(age)
        allowed = AGE_SEED_FAMILY_ALLOWLIST.get(band) or set()
        if not allowed:
            return OPENING_SEED_FAMILIES

        filtered = [seed for seed in OPENING_SEED_FAMILIES if seed.get("family") in allowed]
        return filtered or OPENING_SEED_FAMILIES

    def _age_humour_profile(self, age: Any) -> dict:
        return AGE_HUMOUR_PROFILES.get(self._age_band_key(age), AGE_HUMOUR_PROFILES["5_6"])

    def _select_age_funny_quirk(self, age: Any) -> str:
        profile = self._age_humour_profile(age)
        return random.choice(profile.get("quirks") or FUNNY_QUIRKS)

    def _select_age_plot_humour_event(self, age: Any) -> str:
        profile = self._age_humour_profile(age)
        return random.choice(profile.get("events") or PLOT_HUMOUR_EVENTS)

    def _age_humour_instruction(self, age: Any) -> str:
        profile = self._age_humour_profile(age)
        return profile.get("instruction") or "Use gentle bedtime-safe humour that moves the plot forward."

    def _age_readability_block(self, age: Any) -> str:
        """Oxford-inspired age guidance for readability and story complexity.

        This is guidance only. PillowTales stories are usually read aloud by a
        parent or narrator, so the rules focus on comprehension, vocabulary,
        sentence length, character count, and plot load rather than strict
        independent-reading schemes.
        """
        child_age = self._safe_child_age(age)

        if child_age <= 2:
            profile = """
AGE READABILITY ENGINE — AGE 0-2:
- Write for a baby or toddler being read to, not for independent reading.
- Use very simple, soothing language with familiar words.
- Most sentences should be 3-8 words.
- Use rhythm, repetition, sounds, colours, animals, bedtime routines, cuddles, and simple feelings.
- Use one clear setting only.
- Use one tiny event only.
- Use no more than 1 important helper character.
- Avoid mysteries, twists, clues, complex choices, busy worlds, or multiple locations.
- The story should feel like a calm sensory bedtime story.
"""
        elif child_age <= 4:
            profile = """
AGE READABILITY ENGINE — AGE 3-4:
- Use very clear early bedtime language.
- Most sentences should be 5-10 words.
- Use familiar words, simple actions, and obvious cause and effect.
- Use one small problem that is easy to understand.
- Use no more than 2 important characters.
- Avoid complicated mysteries, layered clues, subplots, or abstract emotional explanations.
- Repetition is good when it helps the child follow the story.
"""
        elif child_age <= 6:
            profile = """
AGE READABILITY ENGINE — AGE 5-6:
- Use simple adventure language that a tired young child can follow.
- Most sentences should be 6-12 words, with only occasional longer sentences when very clear.
- Use familiar vocabulary. Allow only one or two richer storybook words per page, and make their meaning obvious.
- Use one clear goal, one main helper, and one main obstacle.
- Use no more than 3 important characters.
- Funny moments should be visual and easy to understand.
- Do not introduce several new characters, places, objects, and problems on the same page.
- Avoid advanced, poetic, or abstract phrases such as "silent circus of clouds", "silver acrobats", "balanced on moonbeams", or "belonged to the great Star Ringmaster" for this age unless the meaning is immediately concrete.
"""
        elif child_age <= 8:
            profile = """
AGE READABILITY ENGINE — AGE 7-8:
- Use confident child-friendly story language with clear adventure structure.
- Most sentences should be 10-18 words.
- Allow richer vocabulary, but keep the plot easy to track.
- Mysteries may include 2-3 clues, but each clue should be clearly connected.
- Use no more than 4 important characters.
- Emotional moments can be deeper, but should still be shown through actions and dialogue.
"""
        elif child_age <= 10:
            profile = """
AGE READABILITY ENGINE — AGE 9-10:
- Use richer language and stronger story arcs while keeping bedtime clarity.
- Most sentences should be 12-22 words.
- Allow small subplots, deeper choices, and more character motivation.
- Use no more than 5 important characters.
- The child can infer more, but the main goal must remain clear on every page.
- World-building can be more detailed, but must not obscure the story goal.
"""
        else:
            profile = """
AGE READABILITY ENGINE — AGE 11-12:
- Use more sophisticated language, layered feelings, and stronger adventure structure.
- Sentences may be longer when clear and natural.
- Allow more nuanced emotions, motives, and choices.
- Use no more than 5 important characters.
- The story may include a more intricate mystery or challenge, but the reader must never lose the main thread.
- Keep the tone bedtime-safe rather than childish or patronising.
"""

        return profile + """
GENERAL AGE RULE:
- Match the story to the child's age, not to a single default style.
- Younger children need clarity, repetition, and fewer moving parts.
- Older children can handle richer language and deeper emotion, but still need a clear story goal.
"""


    def _age_vocabulary_block(self, age: Any) -> str:
        """Oxford Reading Tree-inspired vocabulary and sentence rhythm guidance.

        This is prompt-only. It does not affect narration, chunking, polling,
        page count, storage, subscriptions, or reader behaviour.
        """
        child_age = self._safe_child_age(age)

        if child_age <= 2:
            profile = """
AGE VOCABULARY ENGINE — AGE 0-2:
- Use mostly first words and high-frequency concrete words: mum, dad, bed, bear, ball, duck, dog, cat, tree, moon, star, cup, hat, home, up, down, in, out, go, see, look, help, hug, sleep.
- Use repeated short phrases and predictable rhythm.
- Prefer sound words and sensory words: pop, splash, tap, hush, warm, soft, big, small.
- Avoid storybook vocabulary that requires explanation: mysterious, ancient, enormous, invisible, discover, adventure, responsibility, promise, courage, patient.
- Avoid figurative language, symbolic lessons, complex magic rules, and abstract feelings.
- Word budget: about 98% familiar words, 2% new words at most.
- Sentence rhythm: very short sentences, one idea per sentence, no clauses joined by commas.
"""
        elif child_age <= 4:
            profile = """
AGE VOCABULARY ENGINE — AGE 3-4:
- Use early-reader, high-frequency words: look, find, help, make, run, jump, play, hold, open, close, happy, sad, brave, kind, big, small, soft, warm, funny, home, garden, toy, bear, rabbit, boat, door, path.
- Introduce only one or two simple storybook words naturally, such as cosy, twinkle, sparkle, whisper, surprise, or adventure.
- Keep new words concrete and easy to understand from the sentence around them.
- Avoid older-child words such as investigate, extraordinary, remarkable, magnificent, responsibility, complicated, mysterious, ancient, impatient, determined, invisible.
- Avoid long noun phrases and poetic descriptions.
- Word budget: about 95% familiar words, 5% new storybook words.
- Sentence rhythm: simple subject-verb-object sentences; short repeated patterns are welcome.
"""
        elif child_age <= 6:
            profile = """
AGE VOCABULARY ENGINE — AGE 5-6:
- Use vocabulary that matches early independent reading and read-aloud comprehension: careful, clever, brave, kind, hidden, lost, found, bright, dark, near, far, bridge, river, forest, castle, dragon, friend, idea, plan, try, choose, fix, share, promise.
- Allow one or two richer words per page when the meaning is clear from context: curious, wondered, discovered, shimmered, puzzled, patient, festival, invitation.
- Prefer clear concrete verbs over adult or abstract verbs: looked, noticed, asked, tried, carried, opened, helped.
- Avoid frequent use of older vocabulary such as investigate, responsibility, extraordinary, magnificent, peculiar, complicated, astonished, remarkable, consequence, tradition, official.
- Do not use adult abstract phrases like “the village had lost hope”, “the courage inside her heart”, or “a symbol of belonging”.
- Word budget: about 90% familiar words, 10% gentle new vocabulary.
- Sentence rhythm: mostly short sentences with one clear action; occasional longer sentence only when easy to read aloud.
"""
        elif child_age <= 8:
            profile = """
AGE VOCABULARY ENGINE — AGE 7-8:
- Use confident chapter-book vocabulary while staying read-aloud friendly: discover, puzzled, curious, careful, ancient, narrow, secret, journey, clue, pattern, practice, promise, festival, invention, message, nervous, proud.
- Allow richer descriptive words, but keep each page anchored in clear action.
- New vocabulary may appear more often, but should still be understandable without stopping the story.
- Avoid language that feels like age 11+ fiction: responsibility as a theme, complicated politics, symbolic identity, consequence-heavy explanations, or sophisticated irony.
- Do not flatten the story into age 3-4 language; the child can handle connected clues, clearer motives, and more varied verbs.
- Word budget: about 82-85% familiar words, 15-18% richer story vocabulary.
- Sentence rhythm: allow joined sentences and short clauses, but avoid dense paragraphs.
"""
        elif child_age <= 10:
            profile = """
AGE VOCABULARY ENGINE — AGE 9-10:
- Use richer middle-grade vocabulary: investigate, mysterious, remarkable, determined, responsibility, confidence, invention, challenge, solution, tradition, invisible, complicated, cautious, suspicious, generous, loyal.
- Allow more precise emotional and problem-solving language, but keep the bedtime tone warm and clear.
- The child can handle clues, motives, promises, fairness, pressure, and two connected problems.
- Do not use babyish repetition or over-simple phrasing unless it is dialogue from a younger character.
- Avoid adult literary prose, heavy symbolism, academic wording, or sentences that feel written for teenagers.
- Word budget: about 75-80% familiar words, 20-25% richer vocabulary.
- Sentence rhythm: varied sentence lengths, including some longer sentences with clear clauses.
"""
        else:
            profile = """
AGE VOCABULARY ENGINE — AGE 11-12:
- Use upper-child vocabulary with nuance: extraordinary, uncertainty, consequence, determination, responsibility, independent, reluctant, thoughtful, remarkable, investigate, evidence, contradiction, confidence, loyalty, forgiveness.
- Allow more layered emotions, stronger dialogue, and more precise descriptions while keeping the story bedtime-safe.
- The story should not sound childish, but it should still sound like children's fiction rather than adult literary prose.
- Avoid talking down to the reader with toddler-level repetition, over-explained feelings, or very basic vocabulary throughout.
- Avoid grim, cynical, romantic, violent, or teen-focused themes.
- Word budget: about 65-70% familiar words, 30-35% richer vocabulary.
- Sentence rhythm: varied and fluent, with longer sentences allowed when clear and natural.
"""

        return profile + """
OXFORD-INSPIRED VOCABULARY RULE:
- Match vocabulary, sentence rhythm, dialogue, and abstraction to the child's age band.
- Do not give age-12 vocabulary to a 3, 4, 5, or 6-year-old.
- Do not give babyish 3-year-old language to a 9, 10, 11, or 12-year-old.
- New words are welcome when they help children grow, but they must be age-suitable and clear from context.
- Prefer natural spoken bedtime language over school-workbook language.
- Vocabulary level should change with age even when the story theme stays the same.
"""

    def _age_cognitive_load_block(self, age: Any) -> str:
        """Control how much the child has to remember at once."""
        child_age = self._safe_child_age(age)

        if child_age <= 2:
            profile = """
AGE COGNITIVE LOAD — AGE 0-2:
- Use one location only.
- Use one tiny goal only.
- Use one helper only.
- Do not use clues, mysteries, hidden motives, choices, journeys, or time jumps.
- Repeat the same comforting object, sound, or action so the story is easy to follow.
"""
        elif child_age <= 4:
            profile = """
AGE COGNITIVE LOAD — AGE 3-4:
- Use one location or one very simple transition.
- Use one clear goal and one simple problem.
- Use no more than one magical rule.
- Do not use layered clues, secret maps, puzzles, or several helpers.
- Each page should be understandable even if the child misses the previous detail.
"""
        elif child_age <= 6:
            profile = """
AGE COGNITIVE LOAD — AGE 5-6:
- Use one main goal from Page 1 to Page 7.
- Use one main helper and one main obstacle.
- Use no more than two magical rules, and explain them through action.
- Do not ask the child to remember several names, objects, places, and rules at once.
- Avoid poetic or symbolic openings that sound beautiful but make the story harder to follow.
"""
        elif child_age <= 8:
            profile = """
AGE COGNITIVE LOAD — AGE 7-8:
- Two connected goals are allowed if the link is clear.
- A small mystery may use 2-3 clues.
- Use up to three magical rules, but remind the reader when they matter.
- Keep the main thread visible on every page.
"""
        elif child_age <= 10:
            profile = """
AGE COGNITIVE LOAD — AGE 9-10:
- A subplot or deeper choice is allowed, but it must support the main goal.
- More detailed worlds are allowed, but do not let world-building bury the action.
- Keep named characters, magical rules, and locations controlled.
"""
        else:
            profile = """
AGE COGNITIVE LOAD — AGE 11-12:
- Layered motives, subplots, and more nuanced choices are allowed.
- The main story thread must still remain easy to summarise in one sentence.
- Do not confuse sophistication with overloaded description.
"""

        return profile + """
GENERAL COGNITIVE LOAD RULE:
- If the story starts to feel busy, remove one character, one object, one location, or one magical rule.
- The child and parent should always understand what problem is being solved now.
"""

    def _age_emotional_conflict_block(self, age: Any) -> str:
        """Match emotion, conflict type, and humour to the child's age."""
        child_age = self._safe_child_age(age)

        if child_age <= 2:
            profile = """
AGE EMOTION AND CONFLICT — AGE 0-2:
- Use simple feelings: happy, sleepy, cosy, surprised, sad, or excited.
- Use tiny conflicts: a lost blanket, sleepy animal, missing cuddle, quiet sound, or bedtime routine.
- Humour should be sound-based or visual: a squeak, a wobble, a tiny sneeze, or a silly hat.
"""
        elif child_age <= 4:
            profile = """
AGE EMOTION AND CONFLICT — AGE 3-4:
- Use simple feelings: happy, sad, scared, excited, worried, proud.
- Use simple conflicts: a lost toy, a sleepy animal, a stuck door, a missing hat, or preparing a small party.
- Humour should be obvious and visual, with no wordplay required.
"""
        elif child_age <= 6:
            profile = """
AGE EMOTION AND CONFLICT — AGE 5-6:
- Use child-friendly feelings: worried, shy, proud, disappointed, brave, patient.
- Use clear conflicts: helping a friend, fixing one problem, delivering one invitation, understanding a misunderstanding, or helping someone try again.
- Avoid abstract conflicts such as a kingdom losing hope, a village forgetting courage, or a world needing balance.
- Humour should be easy to picture and should come from behaviour, not complex dialogue.
"""
        elif child_age <= 8:
            profile = """
AGE EMOTION AND CONFLICT — AGE 7-8:
- Use feelings such as confused, jealous, nervous, determined, left out, or responsible.
- Conflicts may include a small mystery, secret, promise, or choice.
- Humour may include simple misunderstandings or character habits.
"""
        elif child_age <= 10:
            profile = """
AGE EMOTION AND CONFLICT — AGE 9-10:
- Use more layered feelings such as guilt, loyalty, pressure, fairness, or responsibility.
- Conflicts may include competing promises, mistakes, trust, teamwork, or a more involved mystery.
- Humour can be drier, but must remain warm and bedtime-safe.
"""
        else:
            profile = """
AGE EMOTION AND CONFLICT — AGE 11-12:
- Use nuanced feelings such as regret, uncertainty, responsibility, forgiveness, independence, or self-doubt.
- Conflicts may involve motives, consequences, identity, loyalty, or difficult choices.
- Keep the tone emotionally mature but still comforting and bedtime-safe.
"""

        return profile + """
ABSTRACT CONCEPT GUARD:
- For children under 7, show feelings through actions instead of explaining abstract ideas.
- Prefer: "nobody wanted to hang the lanterns" over "the village had lost hope".
- Prefer: "the dragon hid behind the chair" over "the dragon doubted himself".
- Do not use symbolic lessons that require adult interpretation.
"""

    def _age_opening_and_page_rules(self, age: Any) -> str:
        child_age = self._safe_child_age(age)
        if child_age <= 2:
            max_named = 1
        elif child_age <= 4:
            max_named = 2
        elif child_age <= 6:
            max_named = 3
        elif child_age <= 8:
            max_named = 4
        else:
            max_named = 5

        return f"""AGE OPENING AND PAGE RULES:
- By the end of Page 1, the reader must clearly know where the child started, what happened, and why the child is involved.
- Page 1 must not introduce more than one main helper, one problem, and one magical setting idea.
- Do not open with a crowded catalogue of poetic details.
- Each page must have one clear job: arrive, meet, notice, try, choose, solve, or settle.
- Do not introduce a new character, a new place, a new object, and a new problem on the same page.
- Use no more than {max_named} named supporting characters for this age.
- If a page needs explanation to make sense, simplify the page rather than adding more explanation.
"""

    def _age_quality_control_block(self, age: Any) -> str:
        return (
            self._age_cognitive_load_block(age)
            + "\n"
            + self._age_emotional_conflict_block(age)
            + "\n"
            + self._age_opening_and_page_rules(age)
        )

    def _story_clarity_rules(self) -> str:
        return """STORY CLARITY ENGINE:
- Clarity is more important than beautiful writing.
- Prefer clear > clever, simple > poetic, focused > complicated.
- The reader should always know:
  1. Where are we?
  2. Who are we with?
  3. What are we trying to do next?
- Introduce only ONE major new thing per page: one new character, one new place, one new problem, or one new clue.
- Never introduce a new character, a new object, a new place, and a new problem together on the same page.
- Do not introduce several characters, objects, places, and goals in the same paragraph.
- At the start of each page, briefly re-anchor the reader in the current situation.
- Every page should follow clear cause and effect: because A happened, the child did B, which caused C.
- If the story feels busy, remove details rather than add explanations.
- Avoid long chains of events that feel like: this happened, then this happened, then this happened.
- Avoid decorative details that do not help the reader understand the story.
- A tired parent should never have to stop and wonder who a character is, where the story is, or why the child is doing something.
- Before returning the story, silently apply the Lost Test: if a tired parent cannot answer where we are, who we are with, and what we are trying to do next, simplify the page.
"""

    def _character_memory_rules(self) -> str:
        return """PHASE 11 CHARACTER MEMORY AND CALLBACK RULES:
- Treat early details as promises to the reader. If the story introduces a preference, object, phrase, habit, fear, small joke, helper job, or promise, bring at least two of them back later with purpose.
- By Page 2, establish 2-3 reusable memory seeds. Good seeds include: a favourite snack, a repeated phrase, a comfort habit, a helper's job, an unusual object, a small promise, a rule of the place, or a funny misunderstanding.
- Each memory seed must be simple enough for the child's age. Do not create a complicated list of clues for younger children.
- At least one early detail must become useful in the middle of the story, not just decorative.
- At least one early detail must return on the final page as an emotional or visual payoff.
- Characters should remember what other characters said or did. Use this to show friendship, trust, confidence, kindness, patience, or courage through action.
- A side character should not appear, help once, and vanish unless that is the point of the story. If they matter, let their habit, job, phrase, or promise echo later.
- Avoid random callbacks. A callback should either solve a problem, reveal character growth, repair a relationship, create a warm joke, or make the ending feel earned.
- Do not add extra plot just to create callbacks. Reuse what already exists instead.
- The final callback should make a child feel, "I remember that," without the narrator explaining the lesson.
"""

    def _emotional_cohesion_rules(self) -> str:
        return """PHASE 11B EMOTIONAL COHESION RULES:
- Do not merely remember objects. Remember emotions, small conversations, promises, worries, phrases, and choices.
- Select only the 2-4 most meaningful callbacks in the story and make those matter. Do not try to recall every object or detail.
- Give one important supporting character a memorable identity that can be recognised later: a specific job, habit, worry, repeated phrase, unusual tool, or small comic behaviour.
- That supporting character's identity should affect the plot at least once. Avoid generic helpers who only explain the next step.
- Create one simple emotional phrase or idea early in the story, then let the child or helper echo it later through dialogue or action.
- A good emotional callback should sound natural, such as a child remembering a helper's words, repeating a phrase, using a habit, or noticing that someone has changed.
- The child should be the person who makes the decisive emotional or practical choice near the middle or ending. Helpers may guide, but the child should not be carried through the plot.
- Avoid ending by listing everything that happened. The final page should choose one strong emotional callback and one concrete final image.
- Avoid writing "remembered the lesson", "learned that", or "realised". Show memory through what the child says, does, keeps, gives back, or notices.
- If the story includes a repeated phrase, use it no more than twice after the first mention, and make the final use feel earned.
- If a side character changes, show it through behaviour: a worried character pauses, a rushed character waits, a shy character speaks, or a grumpy character helps.
"""

    def _world_logic_rules(self) -> str:
        return """PHASE 11B WORLD LOGIC RULES:
- Give the magical place one simple internal rule, custom, or reason that explains the main problem.
- The rule must be understandable to the child's age and should not require a long explanation.
- Examples: boxes hop when labels are rushed; lanterns only rise after everyone waits for the same breeze; a bakery's biscuits sing only when the oven door is thanked; a post office sorts letters by kindness instead of size.
- Use the world rule to create one obstacle and one solution. Do not add several unrelated magical rules.
- The world should feel handcrafted, not random. Strange details should connect to the setting's jobs, customs, weather, food, animals, or daily routines.
- Prefer one memorable world rule over many magical decorations.
"""

    def _storycraft_rules(self) -> str:
        return """STORYCRAFT QUALITY RULES:
- Write like a skilled children's author rather than a poet. Use clear, warm sentences and avoid over-describing ordinary things.
- Write in simple, natural language that is easy to read aloud to young children. Most sentences should be direct and uncomplicated, saving richer descriptions for occasional special moments.
- Make the story feel like a premium illustrated children's fantasy tale: imaginative, emotionally warm, cinematic, and magical, while remaining original and bedtime-safe.
- Use a classic storybook arc: wonder-filled opening, gentle discovery, small emotional challenge, magical or meaningful helper moment, moral learned through action, and a satisfying peaceful resolution.
- Stories should use a wide variety of story problems.
- Do NOT make most stories about finding, recovering, unlocking, repairing, or returning a magical object.
- The central problem should usually involve people, emotions, relationships, choices, or helping someone rather than recovering or transporting an object.
- At least half of all stories should contain no magical object quest at all.
- The emotional journey should be more important than any magical object.
- If a magical item exists, it should support the story rather than be the main goal.
- No more than 30% of stories should involve magical objects, maps, stones, keys, crystals, or enchanted items.
- Avoid repeatedly using these words unless explicitly requested: silver, moon, moonlit, star, lantern, crystal, glowing, magical key, ancient map.
- The ending should primarily resolve an emotional need rather than simply returning an object.
- Good story themes include friendship, courage, kindness, celebration, misunderstandings, teamwork, learning patience, helping family, discovering talents, and making new friends.
- The story setting should feel vivid, memorable, and specific, like a real storybook world the child can picture immediately.
- The adventure may begin anywhere that suits the theme, not only near a home, bedroom, window, blanket, or bedtime object.
- Use a wide variety of bedtime-safe locations when appropriate: rainforests, river boats, deserts, castles, islands, mountains, oceans, cloud cities, ancient observatories, magical markets, treehouses, hidden valleys, peaceful pirate harbours, underwater palaces, and faraway lands.
- Establish the setting naturally and give the child a simple, believable reason for being there. Avoid stories where the child simply appears in a magical place without context or purpose.
- Every adventure should begin with an Adventure Entry sequence: ordinary world → trigger → transition → reason for the adventure.
- Page 1 should answer three questions before introducing the magical setting: Where was the child? What happened? How did the adventure begin?
- The child should never suddenly appear in a magical forest, floating island, crystal cave, underwater kingdom, or other fantasy location without explanation.
- Begin in an ordinary or understandable place whenever possible: a bedroom, garden, beach, kitchen, campsite, grandparents' house, school, library, or another believable setting.
- Introduce a trigger such as a strange noise, a letter, an animal, a wish, a telescope, footprints, or a hidden path.
- Show a clear transition into the adventure, such as a hidden door, dream, rainbow, tunnel, tree hollow, or secret gate.
- Establish why the adventure matters before the end of page 1: someone needs help, a mystery needs solving, a celebration is beginning, an invitation has arrived, or something important has gone missing.
- Make the setting important to the story rather than background scenery. Distinctive places such as cafés, bakeries, schools, workshops, theatres, trains, and gardens should influence the problem and its solution.
- If the child has a role or identity (princess, pirate, explorer, baker, inventor, astronaut), that role should meaningfully influence the story's events.
- Do not invent unnecessary physical descriptions of the child such as hair colour, eye colour, skin colour, height, clothing, or appearance unless explicitly provided.
- Focus on the child's role, personality, choices, actions, and connection to the setting instead.
- Let the child make choices, notice details, and grow through meaningful scenes. Every page should move the story forward through discovery, decision, challenge, helper, transformation, reflection, or peaceful closure.
- Classic openings such as "Once upon a time..." are allowed when they suit the theme, especially fairytale, princess, castle, dragon, unicorn, or kingdom stories.
- Do not overuse the same opening across stories. Vary openings naturally between classic storybook openings, place-based openings, child-in-role openings, small mysteries, visitors, questions, celebrations, unusual events, and gentle problems.
- Avoid repetitive openings such as "One evening...", "One night...", or "There lived..." when they appear too frequently.
- By the end of page 1, the reader should understand why the child matters in this story, either because they already have a role in the world or because the event, question, discovery, or responsibility now belongs to them.
- Avoid poetic or overly lyrical descriptions and avoid writing every sentence to sound magical. Simple, concrete descriptions are often more memorable than decorative language.
- Avoid repeatedly relying on magical objects as the main story trigger. Sometimes begin with a problem, visitor, animal, mystery, missing item, wish, celebration, question, or natural event instead.
- Avoid sentences where the child already understands the story's lesson before the adventure begins.
- Strongly avoid the words: "gentle", "tiny", "little", "golden", "shimmering", "glowing", "sparkling", "moonlit", "softly", "slowly", and "sleepy". Use them only when absolutely essential to the plot. Never use more than one of these words on a single page.
- Avoid titles containing the words: sleepy, moonlit, little, tiny, golden, sparkling, glowing, or gentle unless they are essential to the story's central idea.
- Avoid giving ordinary objects or scenery human emotions unless it creates a genuinely memorable magical detail. Do not routinely write that flowers wait, trees hope, stars watch, or gardens breathe.
- Do not describe every magical object as shimmering, glowing, sparkling, or golden. Find fresh, specific ways to describe magic and wonder.
- Prefer specific, memorable descriptions over generic magical adjectives. Instead of "glowing pearl", "golden light", or "sparkling fountain", describe what makes the object unusual or memorable.
- Limit repeated use of moon imagery. Avoid filling stories with moonlight, moonbeams, moon-dust, silver leaves, or sleepy night scenes unless the moon is genuinely important to the plot.
- Prefer memorable nouns, actions, and sensory details over decorative adjectives and adverbs. Show magic through what characters see, hear, touch, smell, and do.
- Avoid repeatedly using adverbs ending in "-ly", especially: bossily, excitedly, happily, carefully, suddenly, quickly, and softly. Show emotions and personality through actions and dialogue instead.
- Prefer one memorable detail over many adjectives. A beetle carrying a dew drop is more memorable than a tiny, gentle, shimmering beetle.
- Prefer strong nouns and actions over extra adjectives. If a sentence remains clear and magical after removing an adjective or adverb, remove it.
- Avoid descriptive formulas such as "little + gentle + soft + shimmering + quiet". Vary imagery and vocabulary from story to story.
- At least half of the magical details in a story should come from ordinary things behaving in unexpected ways, such as a teacup collecting raindrops, a staircase made of books, a snail carrying a lantern, a tree that grows bells, or a puddle that remembers songs.
- Include at least one concrete, memorable object or image on each page that a child could easily draw or talk about the next day.



WONDER & ENDINGS ENGINE RULES:
- Endings should avoid relying too often on simple physical rewards such as a marble, crystal, seed, coin, or ordinary keepsake.
- The final reward should preferably be one of:
  • a new friendship,
  • an invitation to return,
  • a promise kept,
  • a small mystery that remains,
  • a magical object with a specific purpose,
  • a helper continuing their work somewhere far away,
  • a relationship repaired,
  • or a quiet sign that another adventure may come.

- If the story gives the child a magical object, make it distinctive and story-specific, not generic.
  Good examples:
  • a compass that points toward kindness,
  • a feather that remembers songs,
  • a bottle containing a tiny sunrise,
  • a key made of moonlight,
  • a bell that rings only when someone needs help,
  • a lantern that stores happy memories,
  • a paper star that follows the child home,
  • a map that redraws itself at bedtime.

- The final paragraph should include one last image of wonder:
  • a light blinking once,
  • a faraway helper still awake,
  • a map changing slightly,
  • a feather moving by itself,
  • a tiny door appearing,
  • a second letter arriving,
  • a star following the child home,
  • or a magical place still quietly existing after the child returns.

- Some stories may end with a small unanswered wonder, as long as the main emotional arc is complete and the child feels safe.
- Avoid overusing endings where someone simply gives the child a marble, crystal, seed, or glowing stone.
- Do not wrap every story up too neatly. A tiny remaining mystery can make the story more memorable while still feeling calm and complete.
- Strengthen magical locations so they feel genuinely fantastical rather than ordinary places with magical adjectives.
  Possible location textures include:
  • floating libraries,
  • upside-down islands,
  • forests of lantern trees,
  • rivers of moonlight,
  • valleys where clouds sleep,
  • bridges woven from stars,
  • gardens inside comets,
  • islands carried by giant turtles,
  • markets that open only when the moon is full,
  • observatories carved into sleeping mountains.

STORY ARCHETYPE RULES:
- Every story must contain a small problem, mystery, challenge, or goal that cannot be solved immediately. The middle of the story should involve at least one meaningful obstacle, choice, or discovery before the resolution.
- The chosen theme must actively drive the plot, not merely decorate the setting. If the theme is dragons, the dragons should shape the adventure through flying, nests, treasure, fire puffs, dragon games, dragon customs, secret caves, sky races, smoke signals, scales, wings, eggs, hoards, or other dragon-specific behaviour.
- For every theme, include at least two theme-specific actions, places, objects, or customs. A pirate story should involve maps, ships, islands, codes, tides, treasure, or crews. A space story should involve planets, telescopes, comets, rockets, constellations, moon stations, or star maps. A princess/kingdom story should involve courts, gardens, castles, royal duties, festivals, crowns, bridges, or quests.
- Avoid slice-of-life stories where characters simply walk, eat, tidy, water plants, and talk unless those actions directly solve the story's central problem.
- Food, chores, gardens, cafés, bakeries, and cosy routines may appear, but they must support the adventure rather than replace it.
- Children should actively solve, discover, repair, rescue, deliver, protect, choose, test, translate, guide, or uncover something during the adventure.
- The story should contain one clear turning point around the middle pages where the child’s action changes the outcome.
- Keep the stakes bedtime-safe but meaningful: a lost friend needs help, a special place needs fixing, a promise must be kept, a celebration might be missed, a shy creature needs courage, a map is incomplete, or a small magical task must be finished before nightfall.
- Around the middle of the story, include one genuine obstacle, complication, or unexpected problem that briefly makes success uncertain. The obstacle should require the child to make a choice, solve something, or help someone in a new way.
- Around the middle of the story, include an obstacle that cannot be solved immediately. The child's first idea may fail, reveal new information, or create a new challenge. Success should briefly feel uncertain before the child discovers another way forward.

- At least one obstacle should involve a meaningful choice, such as choosing between two paths, deciding who to help first, giving something up, solving a puzzle, trusting someone unexpected, or trying a second idea after the first one does not work.
- Whenever possible, let the child's first solution fail or only partly succeed, revealing new information and requiring a different approach before the resolution.
- Whenever possible, include a moment where the child must make a choice, such as choosing between two paths, deciding who to help first, giving up an important object, solving a riddle, trusting an unfamiliar creature, or using creativity instead of strength.
- Avoid using the same story structure repeatedly. Not every story should involve finding a creature, receiving a gift, or returning home immediately after the adventure.
- Different stories may involve solving a mystery, completing a delivery, following clues, discovering a hidden place, preparing for a celebration, helping two characters reconcile, searching for a missing object, repairing something magical, escaping a changing environment, or uncovering a secret.
- Vary how stories begin. Some stories may start with a strange sound, an unusual visitor, a mysterious object, an invitation, an unexpected event, a problem already in progress, or a surprising discovery.
- Strongly avoid repetitive AI phrases and wording such as: "rhythmic humming", "steady humming", "safe, happy, and warm", "closed its eyes", "floated peacefully", "softly humming", "shivering alone", "calm heart", "long breath", "everything was quiet and safe", and "ready for sleep". Also avoid repeatedly ending with generic marbles, crystals, glowing stones, seeds, or coins. Use fresh language and concrete actions instead.

STORY MEMORY RULES:
PICTURE-BOOK MOMENT RULE:
- Every story MUST contain one unforgettable picture-book illustration moment.
- Include one scene that a child could easily draw or describe tomorrow.

Examples:
• a dragon wearing oven gloves
• a whale carrying lanterns
• a staircase made of books
• a fox painting stars
• a tree growing bells
• a teacup floating on a cloud
• a pirate ship made of pillows
• a pony wearing a flower crown
- Every story MUST include at least one supporting character who:
  • has a job,
  • has a funny habit,
  • says something unusual,
  • or owns an unusual object.
- Supporting characters should feel specific and memorable rather than generic helpers.
- Every story must include one memorable object that a child could draw, describe, or ask about tomorrow. The object should matter to the story, not just appear as decoration.
- Every story must include one memorable place that feels different from an ordinary setting and shapes what happens there.
- Every story must include one moment of kindness, humour, surprise, courage, patience, or reassurance that changes what happens next.
- At least one scene MUST feel like the front cover of a premium children's picture book.
- The image should be concrete, visual, and easy to remember.
- The hero should feel like a real child with habits and preferences.
- Give the hero one memorable quirk and one comfort habit.
- Include at least TWO gentle giggle moments in every story.
- At least one giggle moment must directly change the plot.
- One should come from a character's behaviour or habit.
- One should come from dialogue or a misunderstanding.
- Humour should come from behaviour, habits, misunderstandings, or personality rather than jokes.
- The humour should feel warm, character-driven, and memorable, in the tradition of classic bedtime picture books.
- Important side characters should have distinctive habits, jobs, sayings, or behaviours.

GENTLE HUMOUR RULES:
- Include 1-2 genuine giggle moments.
- Supporting characters may:
  • misunderstand instructions
  • take things too literally
  • collect unusual objects
  • say unexpected things
  • become distracted by something silly
- Humour should never be noisy, mean, or embarrassing.
- Avoid modern jokes, sarcasm, memes, or punchlines.
- Funny moments must move the story forward and create consequences.
- At least one personality trait or quirk should help solve the story problem.
- Avoid describing characters only with adjectives.
- Show personality through actions and repeated behaviours.
- Give important side characters small personalities or jobs, such as a dragon who collects teacups, a fox who paints stars, a snail who delivers letters, a rabbit who draws maps, a bear who bakes midnight pies, or a pirate who sorts buttons.
- Extra supporting character examples: a dragon who alphabetises biscuits, a rabbit who wears three scarves, a pirate who is scared of seagulls, a bear who keeps forgetting where he put his hat, or a fox who paints vegetables instead of pictures.
- After introducing an important character, avoid repeating their name in every paragraph.
- Vary references naturally using descriptions, species, titles, occupations, or relationships.
  Examples: Barnaby -> the badger, the Burrow Warden, her new friend; Princess Elara -> the young princess, the gardener's daughter; Captain Moss -> the old sailor, the map keeper.
- Do not begin consecutive paragraphs with the same character's name. Vary sentence openings naturally.
- Generate a wide variety of character names. Avoid repeatedly using common storybook names such as Barnaby, Hazel, Pip, Fern, Willow, Bramble, Luna, Oliver, Poppy, Archie, Daisy, or Jasper unless the name is genuinely fresh in context.
- Prefer fresh, memorable, and varied names that fit the setting.
- Do not rely on generic labels such as guardian, keeper, magical creature, mysterious animal, wise helper, or friendly guide unless the character also has a specific personality, relationship, or job.
- Strengthen the story promise beyond simply finding a clue. Examples of stronger bedtime-safe promises: deliver the last moon biscuit, repair the rainbow bridge, find the missing laugh, return a borrowed song, wake tomorrow's sunrise, rescue a lost recipe, or help a shy dragon practise a tiny roar.
- Make at least one story detail something a child might say again the next day, such as “the teacup dragon”, “the bell tree”, “the map rabbit”, “the biscuit moon”, or “the boat made from folded maps”.
- The final page should contain an emotional callback to an earlier object, place, promise, or supporting character.
- The final paragraph should remind the reader of something encountered earlier.
- End with one final magical image and avoid generic sleep endings.
- The ending should feel emotionally earned and memorable.
- The ending should feel emotionally earned and memorable, as if a child might ask for this particular story again tomorrow.
- Never directly state a character's thoughts, wishes, intentions, realizations, or feelings. Do not write: "she wanted...", "he hoped...", "she realised...", "he knew...", "she felt...", "he understood...", or "she decided...". Instead, show these through:
  - actions
  - dialogue
  - expressions
  - choices
  - consequences
- Trust the reader to infer emotions and lessons. Do not explain what the child should feel.
- Never explain the story's moral before the child discovers it through events. Avoid phrases such as "learned that", "needed to teach", "realized that", "the magic of sharing", or similar explanations of the lesson.
- Avoid repeatedly relying on titled magical helpers such as keepers, guardians, tenders, weavers, or shepherds. Use them only when they feel genuinely original and important to the story.
- Important characters should feel specific and memorable, with their own personalities, relationships, or jobs, rather than generic fantasy roles.
- Let readers discover the moral through actions and consequences rather than narration.
- Write each story as though it were written by a different storyteller. Vary settings, magical details, sentence rhythms, imagery, helpers, and vocabulary so that stories do not feel generated from the same formula.
- Avoid repeatedly using "steady", "calm heart", "long breath", "waited", "inch by inch", "slowly", "quiet and safe", or "rhythm of breathing". Choose fresh expressions and actions instead.
- Most of the story should focus on wonder, discovery, character action, and gentle adventure. Let the final third gradually become calmer and more reflective instead of making every page feel sleepy.
- Make magical worlds internally consistent. If a character has a title or role, give it simple context and give important characters clear relationships or purposes rather than labels alone.
- Avoid flat summaries. Write immersive scenes that feel read-aloud, memorable, and emotionally rewarding.
- Keep the mood safe for bedtime: no danger, no frightening villains, no peril, and no sadness-heavy ending.
- Do not copy or imitate any existing franchise, character, studio, film, song, or copyrighted story world.
- The child should face a meaningful but gentle problem that is resolved through the chosen moral, shown through actions and consequences rather than explanation.
- The story should still work even if the word for the moral never appears.
- Vary the moral lessons across stories. Avoid repeating sharing, kindness, helping, or teamwork as the default moral.
- If the chosen moral is broad, make it specific through the story situation.
- Avoid turning the moral into a lecture. The child should understand it because the story outcome feels emotionally true.
- Keep the moral age-appropriate, hopeful, and reassuring for bedtime.
- Give each story one memorable image, object, sound, or moment that a child might mention again tomorrow.
- Let the ending gently return the child toward safety, comfort, and sleep."""

    def _select_opening_seed(self, request: GenerateStoryRequest) -> dict:
        """Select a place-first opening locally before Gemini is called.

        This is intentionally instant: no extra AI call, no database lookup, and
        no narration/playback impact. The opening gives the child a clear place
        to enter before the magical event begins.
        """
        child = request.childName or "the child"
        language_code = (request.storyLanguageCode or "en").lower()[:2]
        seed = random.choice(self._seed_pool_for_age(request.age))
        template = seed.get(language_code) or seed["en"]
        return {
            "family": seed.get("family", "place_entry"),
            "sentence": template.replace("{childName}", child),
        }

    def _opening_transition_rule(self, opening_family: str) -> str:
        return (
            f"Use the '{opening_family}' place-entry idea as the story doorway. "
            "Stay grounded in that place and move into one clear first action. "
            "Make the magical trigger feel caused by something the child is already doing there."
        )

    def _localized_opening_sentence(self, request: GenerateStoryRequest) -> str:
        # Compatibility wrapper for older internal calls.
        return self._select_opening_seed(request)["sentence"]

    def _language_style_block(self, language_code: Optional[str]) -> str:
        language_code = (language_code or "en").lower()[:2]
        if language_code == "fr":
            return """
FRENCH STORY STYLE — MATCH ENGLISH QUALITY:
- Write as a native French bedtime storyteller, not as a translation from English.
- Keep the same emotional shape as the English stories: warm opening, gentle discovery, child-led kindness, magical wonder, peaceful ending.
- Use simple, beautiful French that sounds natural when read aloud to a child aged 4-8.
- Avoid formal, academic, old-fashioned, abstract, or overly literary phrasing.
- Prefer short flowing sentences, soft sensory details, and intimate bedtime warmth.
- Do not make the French version stranger, darker, more philosophical, or more complicated than the English tone.
- The story should feel tender, magical, cosy, clear, and reassuring.
"""
        if language_code == "es":
            return """
SPANISH STORY STYLE — STRICT CASTILIAN/SPAIN ONLY:
- Write in natural Spanish from Spain (castellano peninsular), like a parent in Spain gently reading a bedtime story aloud.
- The result must sound originally written in Spain, not Latin American, neutral-dubbed, AI-translated, or Disney-dub Spanish.
- Use Spain bedtime cadence: warm, close, calm, simple, and softly musical. Sentences should feel spoken naturally at bedtime, not over-described.
- Prefer natural Spain wording: cuento, habitación, manta, arroparse, cariño, dormir, sueños, despacio, suave, tranquilo/a, pequeño/a, con cuidado, poco a poco, una pizca de magia, al lado de la cama.
- Avoid Latin American vocabulary and rhythm. Do not use: computadora, carro, lindo/linda as the main adjective, platicar, celular, ustedes for family/child address, chiquito/chiquita, calientito/calientita, colorido as a filler adjective.
- Avoid neutral/Latin-American-style diminutives for magical things. Do not write “dragoncito”, “lucecita”, “estrellitas”, “chispitas”, or similar if “pequeño dragón”, “luz suave”, “pequeñas estrellas”, or “destellos suaves” sounds more Spain-native.
- Avoid phrases that feel translated or over-dramatic, such as “la noche quería contarle un secreto” repeated too often, “pura curiosidad”, or “comienzo de una aventura muy especial” unless it sounds fully natural in context.
- Use tú-style warmth and Spain-family intimacy. Do not use “ustedes” for the child/family voice.
- Keep the same emotional shape as the English stories: warm opening, gentle discovery, child-led kindness, magical wonder, peaceful ending.
- Keep it child-friendly and read-aloud: concrete images, gentle feelings, and cosy actions. Do not sound textbook, stiff, theatrical, or cartoon-dubbed.
- If choosing between a generic Spanish phrase and a Spain-native phrase, always choose the Spain-native phrase.
"""
        if language_code == "it":
            return """
ITALIAN STORY STYLE:
- Write with warm, natural Italian suitable for young children.
- Avoid overly formal, stiff, or literal phrasing.
- Use gentle, musical bedtime rhythm and soft magical imagery.
- The story should feel cozy, tender, dreamy, comforting, and originally written in Italian.
"""
        if language_code == "de":
            return """
GERMAN STORY STYLE:
- Write with warm, natural German suitable for young children.
- Avoid overly formal, stiff, academic, or literal phrasing.
- Use gentle bedtime rhythm, cozy imagery, and emotionally comforting language.
- The story should feel soft, magical, reassuring, read-aloud friendly, and originally written in German.
"""
        return """
ENGLISH STORY STYLE:
- Use warm, premium British bedtime storytelling with soft rhythm, clear emotion, and child-friendly magic.
- Keep the story gentle, imaginative, cosy, and easy to read aloud.
"""

    def _first_page_language_style_block(self, language_code: Optional[str]) -> str:
        """Compact language guidance for the speed-critical Page 1 call.

        The full language style block is intentionally not used here because
        Page 1 must return quickly. Richer language rules remain in the
        background continuation prompts.
        """
        language_code = (language_code or "en").lower()[:2]
        if language_code == "es":
            return "Write natural Spanish from Spain (castellano), warm, simple, and read-aloud friendly. Avoid Latin-American or stiff translated phrasing."
        if language_code == "fr":
            return "Write natural French bedtime prose, warm, clear, and read-aloud friendly. Avoid stiff translation or overly literary phrasing."
        if language_code == "it":
            return "Write natural Italian bedtime prose, warm, clear, and read-aloud friendly. Avoid stiff or literal phrasing."
        if language_code == "de":
            return "Write natural German bedtime prose, warm, clear, and read-aloud friendly. Avoid stiff or academic phrasing."
        return "Write warm, clear, read-aloud English bedtime prose. Keep sentences direct and child-friendly."

    def _build_prompt(self, request: GenerateStoryRequest, companion: Optional[dict]) -> str:
        language_name = SUPPORTED_LANGUAGES.get(request.storyLanguageCode, 'English')
        language_code = (request.storyLanguageCode or "en").lower()[:2]
        effective_theme = (
            request.customTheme
            if request.customTheme
            else self._localized_theme_label(request.theme, language_code)
        )
        localized_companion = self._localized_companion(companion, language_code)
        archetype = self._select_story_archetype()
        archetype_block = self._story_archetype_block(archetype)
        emotional_theme = self._select_emotional_story_type()
        emotional_block = self._emotional_story_block(emotional_theme)
        character_trait = self._select_character_trait()
        funny_quirk = self._select_age_funny_quirk(request.age)
        personality_humour_block = self._personality_humour_block(character_trait, funny_quirk, request.age)

        companion_line = self._no_companion_required_text(language_code)
        if localized_companion:
            companion_line = (
                f"Include {localized_companion['name']} naturally in the story. "
                f"They are described as: {localized_companion['description']}. "
                "Make them warm, helpful, and bedtime-appropriate."
            )

        family_characters = request.characters or []
        if family_characters:
            character_lines = []
            for character in family_characters[:3]:
                relationship = self._localized_relationship_label(character.relationship, language_code)
                character_lines.append(f"- {character.name} ({relationship})")
            characters_block = '\n'.join(character_lines)
            character_instruction = (
                "Include these family members, friends, or pets naturally in the story if possible. "
                "Make sure each named character appears clearly at least once without overwhelming the bedtime tone. "
                "For pets, use ONLY the pet name and animal type provided. Do not invent colour, breed, markings, size, collar, eye colour, or other physical details. "
                "If a pet is named Luna (cat), write about Luna the cat, not a fluffy/black/orange/striped cat unless the parent explicitly provided that detail.\n"
                f"{characters_block}"
            )
        else:
            character_instruction = self._no_extra_characters_required_text(language_code)

        # Standard bedtime narration target:
        # The product now uses one optimal story length. Age controls complexity;
        # duration is kept as an internal compatibility field only.
        target_pages = "7"
        paragraphs_per_page = "2"
        sentence_range = "5-7"
        target_words = "950-1100"
        min_words_per_page = "125"
        ideal_words_per_page = "135-155"
        max_words_per_page = "170"
        pacing_note = (
            "Create a substantial but calm bedtime story suitable for an approximately eight-minute bedtime experience. "
            "Do not compress the plot into a short summary; let each page include a gentle, memorable story moment. "
            "Keep page lengths balanced so the reader does not feel that some pages are rushed while others are overloaded."
        )

        language_style_block = self._language_style_block(request.storyLanguageCode)

        return f"""You are a premium children's bedtime storyteller.

IMPORTANT LANGUAGE RULE:
- The ENTIRE story MUST be written ONLY in {language_name}.
- Do NOT use English unless the language is English.
- Do NOT mix languages.
- All narration, title, and dialogue MUST be in {language_name}.
- Write naturally for native-speaking children in {language_name}.
- The story must feel like it was originally written in {language_name}, not translated from English.
- Use warm, magical, emotionally comforting bedtime storytelling.
- Avoid overly formal, academic, rigid, or literal phrasing.
- Use natural rhythm and gentle emotional pacing suitable for read-aloud bedtime stories.
{language_style_block}

STORY REQUIREMENTS:
- Child name: {request.childName}
- Age: {request.age}
- Theme: {effective_theme}
- Moral: {request.moral}
- Calm level: {request.calmLevel}
- Tone: warm, magical, calming, bedtime-safe
- Use simple, natural language suitable for reading aloud
- No scary content
- No rushed ending
- Do NOT write "The end"
- End peacefully and softly

{self._storycraft_rules()}
{self._oxford_inspired_age_profile_block(request.age)}
{self._story_clarity_rules()}
{self._character_memory_rules()}
{self._emotional_cohesion_rules()}
{self._world_logic_rules()}
{self._age_readability_block(request.age)}
{self._age_vocabulary_block(request.age)}
{self._age_quality_control_block(request.age)}

{archetype_block}
{emotional_block}
{personality_humour_block}
LENGTH AND STRUCTURE REQUIREMENTS (STRICT PERFORMANCE RULES):
- EXACTLY {target_pages} pages. Do not return more or fewer pages.
- EACH page should contain exactly {paragraphs_per_page} gentle paragraphs.
- EACH page should contain approximately {sentence_range} bedtime-friendly sentences in total.
- TOTAL story length MUST be approximately {target_words} words.
- PAGE BALANCE IS STRICT: each page should normally be {ideal_words_per_page} words.
- No page should be under {min_words_per_page} words unless it is the final page and the emotional ending is already complete.
- No page may exceed {max_words_per_page} words.
- Do not create one very long page followed by a very short page. Distribute story beats evenly across all 7 pages.
- Use simple, natural sentences suitable for spoken bedtime narration.
- Do not make pages too short. Avoid summarising scenes in only one or two sentences.
- Every page must move the story forward gently and include one memorable story beat.
- The moral should be discovered through the child's actions, not explained like a lesson.
- The final page must end peacefully and softly, with a clear callback to an object, place, helper, or promise from earlier in the story.
- {pacing_note}

COMPANION:
- {companion_line}

CHARACTERS:
- {character_instruction}

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON:
{{"title": "...", "pages": ["page 1 text", "page 2 text", "page 3 text"]}}

OUTPUT QUALITY RULES:
- Return a complete bedtime story, not an outline
- Before returning JSON, silently check that all 7 pages are similar in length and each page contains a complete story beat.
- Do not include notes, markdown, or explanations outside the JSON
- Keep the story calm and readable, but do not make it too short.
- If unsure, prioritise balanced page length and the requested narration length while staying bedtime-safe.
- The JSON pages array must contain exactly {target_pages} strings."""


    def _intended_page_count(self, request: GenerateStoryRequest) -> int:
        return 7

    def _clean_json_response(self, response_text: str) -> Dict[str, Any]:
        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())

    def _extract_balanced_json_object(self, response_text: str) -> Optional[str]:
        """Extract the first balanced JSON object from a model response.

        Gemini sometimes returns valid JSON surrounded by fences or whitespace.
        This repair is intentionally conservative: it only succeeds when braces
        balance cleanly. It does not invent missing story text or close partial
        strings, so it cannot convert a truly truncated response into a fake
        page.
        """
        if not response_text or not isinstance(response_text, str):
            return None

        text = response_text.strip()
        if text.startswith('```json'):
            text = text[7:].strip()
        if text.startswith('```'):
            text = text[3:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()

        start = text.find('{')
        if start == -1:
            return None

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if escaped:
                escaped = False
                continue
            if ch == '\\':
                escaped = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        return None

    def _parse_json_response_with_repair(self, response_text: str, label: str) -> Dict[str, Any]:
        try:
            return self._clean_json_response(response_text)
        except Exception as primary_exc:
            extracted = self._extract_balanced_json_object(response_text)
            if extracted:
                try:
                    parsed = json.loads(extracted)
                    print(f"[DEBUG] json_repair_success label={label} chars={len(extracted)}")
                    return parsed
                except Exception as repair_exc:
                    print(f"[DEBUG] json_repair_failed label={label} error={repair_exc}")
            print(f"[DEBUG] json_parse_failed label={label} error={primary_exc}")
            raise

    def _salvage_complete_pages_from_partial_json(self, response_text: str, label: str) -> list[str]:
        """Recover fully closed page strings from a truncated {"pages": [...]} response.

        Gemini occasionally stops mid-string. This helper only salvages strings
        that were completely closed before truncation. It never invents text,
        never closes an unfinished sentence, and never fabricates JSON. The
        caller can publish the complete pages and ask Gemini only for what is
        still missing.
        """
        if not response_text or not isinstance(response_text, str):
            return []

        text = response_text.strip()
        if text.startswith('```json'):
            text = text[7:].strip()
        if text.startswith('```'):
            text = text[3:].strip()
        if text.endswith('```'):
            text = text[:-3].strip()

        pages_key = text.find('"pages"')
        if pages_key == -1:
            return []
        array_start = text.find('[', pages_key)
        if array_start == -1:
            return []

        pages: list[str] = []
        idx = array_start + 1
        n = len(text)
        decoder = json.JSONDecoder()

        while idx < n:
            while idx < n and text[idx] in ' \r\n\t,':
                idx += 1
            if idx >= n or text[idx] == ']':
                break
            if text[idx] != '"':
                # If the next value is not a JSON string, stop conservatively.
                break
            try:
                value, end_idx = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                # Most likely an unterminated final page. Keep earlier pages only.
                break
            if not isinstance(value, str):
                break
            pages.append(value)
            idx += end_idx

        cleaned_pages = postprocess_story_pages(pages)
        if cleaned_pages:
            print(
                f"[DEBUG] json_partial_pages_salvaged label={label} "
                f"count={len(cleaned_pages)} response_chars={len(response_text)}"
            )
        return cleaned_pages

    async def _generate_remaining_pages_batch(
        self,
        request: GenerateStoryRequest,
        companion: Optional[dict],
        title: str,
        working_pages: list[str],
        batch_count: int,
        next_page_number: int,
        attempt_label: str,
    ) -> list[str]:
        prompt = self._build_remaining_pages_prompt(
            request=request,
            companion=companion,
            title=title,
            existing_pages=working_pages,
            remaining_page_count=batch_count,
            next_page_number=next_page_number,
        )
        print(
            f"[PERF] remaining_pages_batch prompt chars={len(prompt)} "
            f"next_page={next_page_number} count={batch_count} attempt={attempt_label}"
        )
        t_gemini = time.time()
        response = await asyncio.to_thread(
            self.model.generate_content,
            prompt,
            generation_config=REMAINING_PAGES_GENERATION_CONFIG,
        )
        print(
            f"[PERF] remaining_pages_batch Gemini took {time.time() - t_gemini:.2f}s "
            f"next_page={next_page_number} count={batch_count} attempt={attempt_label}"
        )
        self._log_gemini_response_metadata(
            f"remaining_pages_next_{next_page_number}_count_{batch_count}_{attempt_label}",
            response,
        )

        response_text = getattr(response, 'text', None)
        if not response_text or not isinstance(response_text, str):
            raise ValueError(f'Failed to generate remaining pages batch attempt={attempt_label}')

        label = f"remaining_pages_next_{next_page_number}_count_{batch_count}_{attempt_label}"
        try:
            story_data = self._parse_json_response_with_repair(response_text, label)
        except Exception as parse_exc:
            salvaged_pages = self._salvage_complete_pages_from_partial_json(response_text, label)[:batch_count]
            if salvaged_pages:
                print(
                    f"[PERF] remaining_pages_batch salvaged complete pages "
                    f"next_page={next_page_number} requested={batch_count} salvaged={len(salvaged_pages)} "
                    f"attempt={attempt_label}"
                )
                return salvaged_pages

            raw_preview = str(response_text or '')[:1200]
            print(
                f"[DEBUG] remaining_pages raw Gemini response parse_failed "
                f"attempt={attempt_label} next_page={next_page_number} count={batch_count} error={parse_exc}"
            )
            print(
                f"[DEBUG] remaining_pages raw Gemini response preview "
                f"attempt={attempt_label} value={raw_preview!r}"
            )
            raise

        if not isinstance(story_data, dict) or 'pages' not in story_data:
            raise ValueError(f'Invalid remaining-pages batch format returned by AI attempt={attempt_label}')

        batch_pages = postprocess_story_pages(story_data.get('pages', []))[:batch_count]
        if not batch_pages:
            raise ValueError(
                f'Remaining generation produced no usable pages in batch attempt={attempt_label}'
            )
        if len(batch_pages) < batch_count:
            print(
                f"[PERF] remaining_pages_batch accepted partial valid pages "
                f"next_page={next_page_number} requested={batch_count} got={len(batch_pages)} "
                f"attempt={attempt_label}"
            )
        return batch_pages

    async def _generate_remaining_pages_with_recovery(
        self,
        request: GenerateStoryRequest,
        companion: Optional[dict],
        title: str,
        working_pages: list[str],
        preferred_batch_count: int,
    ) -> list[str]:
        """Generate continuation pages, reducing batch size on parse/truncation failure.

        This prevents the app becoming stranded at 4/7 if Gemini truncates a
        3-page continuation. It is background-only and preserves Page 1 speed.
        """
        next_page_number = len(working_pages) + 1
        attempts: List[tuple[int, str]] = [(preferred_batch_count, 'primary')]
        for size in BACKGROUND_RECOVERY_BATCH_SIZES:
            if size < preferred_batch_count and size > 0:
                attempts.append((size, f'recovery_{size}page'))

        last_exc: Optional[Exception] = None
        for batch_count, attempt_label in attempts:
            if batch_count > preferred_batch_count:
                continue
            try:
                return await self._generate_remaining_pages_batch(
                    request=request,
                    companion=companion,
                    title=title,
                    working_pages=working_pages,
                    batch_count=batch_count,
                    next_page_number=next_page_number,
                    attempt_label=attempt_label,
                )
            except Exception as exc:
                last_exc = exc
                print(
                    f"[PERF] remaining_pages_batch attempt failed story_title={title!r} "
                    f"next_page={next_page_number} count={batch_count} attempt={attempt_label} error={exc}"
                )

        raise ValueError(
            f'Remaining pages recovery failed at page {next_page_number}: {last_exc}'
        )

    def _language_and_character_blocks(self, request: GenerateStoryRequest, companion: Optional[dict]) -> Dict[str, str]:
        language_name = SUPPORTED_LANGUAGES.get(request.storyLanguageCode, 'English')
        language_code = (request.storyLanguageCode or "en").lower()[:2]
        effective_theme = (
            request.customTheme
            if request.customTheme
            else self._localized_theme_label(request.theme, language_code)
        )
        localized_companion = self._localized_companion(companion, language_code)

        companion_line = self._no_companion_required_text(language_code)
        if localized_companion:
            companion_line = (
                f"Include {localized_companion['name']} naturally in the story. "
                f"They are described as: {localized_companion['description']}. "
                "Make them warm, helpful, and bedtime-appropriate."
            )

        family_characters = request.characters or []
        if family_characters:
            character_lines = []
            for character in family_characters[:3]:
                relationship = self._localized_relationship_label(character.relationship, language_code)
                character_lines.append(f"- {character.name} ({relationship})")
            characters_block = '\n'.join(character_lines)
            character_instruction = (
                "Include these family members, friends, or pets naturally in the story if possible. "
                "Make sure each named character appears clearly at least once without overwhelming the bedtime tone. "
                "For pets, use ONLY the pet name and animal type provided. Do not invent colour, breed, markings, size, collar, eye colour, or other physical details. "
                "If a pet is named Luna (cat), write about Luna the cat, not a fluffy/black/orange/striped cat unless the parent explicitly provided that detail.\n"
                f"{characters_block}"
            )
        else:
            character_instruction = self._no_extra_characters_required_text(language_code)

        return {
            'language_name': language_name,
            'effective_theme': effective_theme,
            'companion_line': companion_line,
            'character_instruction': character_instruction,
        }

    def _first_page_age_prompt_rules(self, age: Any) -> str:
        """Compact age-specific Page 1 rules.

        Page 1 is the speed-critical path. Do not include the full age,
        vocabulary, cognitive-load, and emotional engines here. Those richer
        rules still apply to the background continuation/full-story prompts.
        """
        child_age = self._safe_child_age(age)
        if child_age <= 2:
            return """AGE-SPECIFIC PAGE 1 RULES — 0-2:
- Write for a baby or toddler being read to.
- Use very short, soothing sentences of about 3-8 words.
- Use familiar concrete words, soft sounds, bedtime objects, animals, colours, cuddles, and simple feelings.
- Use one familiar place, one tiny discovery, and at most one helper.
- Do not use mysteries, clues, choices, busy worlds, abstract lessons, or complex magic.
- The page should feel calm, sensory, repetitive, and easy to follow."""
        if child_age <= 5:
            return """AGE-SPECIFIC PAGE 1 RULES — 3-5:
- Use clear early-childhood bedtime language.
- Most sentences should be 5-10 words.
- Use one clear place, one simple magical discovery, one small problem, and no more than one helper.
- Make cause and effect obvious: something changes, the child notices, then the child gently joins in.
- Use concrete words and simple visual humour only if it fits naturally.
- Avoid layered clues, several characters, abstract feelings, and poetic description."""
        if child_age <= 8:
            return """AGE-SPECIFIC PAGE 1 RULES — 6-8:
- Use confident child-friendly adventure language.
- Most sentences should be 8-16 words.
- Establish one clear goal, one main helper or clue, and one memorable setting detail.
- A small mystery, delivery, rescue, repair, celebration, or misunderstanding may begin here.
- Include one reusable memory seed that can matter later.
- Keep the page clear; do not introduce several names, objects, places, and rules together."""
        return """AGE-SPECIFIC PAGE 1 RULES — 9-12:
- Use richer but still bedtime-safe language.
- Sentences may be more varied, but the opening must remain clear and quick to understand.
- Establish a stronger hook, one clear story problem, and one specific reason the child is involved.
- Allow a more interesting mystery, choice, responsibility, or world rule, but keep only one main thread.
- Include one emotionally useful memory seed, phrase, promise, or character detail.
- Avoid dense world-building, adult literary prose, or over-complicated setup."""

    def _build_first_page_prompt(self, request: GenerateStoryRequest, companion: Optional[dict]) -> str:
        blocks = self._language_and_character_blocks(request, companion)

        # Phase 11C: Page 1 prompt is now age-specific and deliberately compact.
        # Page 1 is the speed-critical path for narration. Do not include the
        # full Story Bible, full age engines, archetypes, emotional engine, or
        # character/personality engine here. Those remain in background
        # continuation generation where they cannot delay first narration.
        opening_seed = self._select_opening_seed(request)
        opening = opening_seed["sentence"]
        opening_transition_rule = self._opening_transition_rule(opening_seed["family"])
        language_code = (request.storyLanguageCode or "en").lower()
        is_english = language_code == "en"
        age_rules = self._first_page_age_prompt_rules(request.age)

        if is_english:
            page_length_rule = "500-650 characters total, including spaces. Do not exceed 650 characters."
            sentence_rule = "4-6 calm, read-aloud sentences"
        else:
            page_length_rule = "475-625 characters total, including spaces. Do not exceed 625 characters."
            sentence_rule = "4-6 calm, read-aloud sentences"

        return f"""You are writing Page 1 of a premium children's bedtime story.

IMPORTANT LANGUAGE RULE:
- Write ONLY in {blocks['language_name']}.
- Do NOT mix languages.
{self._first_page_language_style_block(request.storyLanguageCode)}

STORY CONTEXT:
- Child: {request.childName}, age {request.age}
- Theme: {blocks['effective_theme']}
- Moral: {request.moral}
- Calm level: {request.calmLevel}

{age_rules}

OXFORD-INSPIRED PAGE 1 CALIBRATION:
- Match this opening to the child's developmental reading/listening level: age should affect sentence length, vocabulary, dialogue, humour, emotional simplicity, and plot load.
- This is only guidance; do not copy Oxford Reading Tree content, characters, style, or wording.

PAGE 1 JOB:
- Show where the child is, what the child is already doing there, what unusual thing happens, and why the child must join in.
- The adventure must grow naturally from the place or activity, not feel like the child is suddenly dropped into a magical setting.
- Use one clear trigger/discovery and one clear story promise.
- Include one reusable detail for later: object, phrase, habit, helper detail, promise, or simple world rule.
- The child must actively notice, ask, choose, follow, help, or begin solving.
- Do not resolve the story yet. No danger, villains, appearance details, or heavy world-building.
- Prefer curiosity/action/dialogue over explanation.

NATURAL OPENING CONTRACT:
- Do NOT begin with "Suddenly", "One day", "One night", "One evening", or "There once was".
- Before the magic appears, make the ordinary reason clear: the child is playing, helping, visiting, building, reading, drawing, walking, tidying, waiting, or exploring.
- The first magical event should be connected to that ordinary action.
- Avoid teleport-style openings where the child simply finds themselves somewhere new.
- Keep the setup quick, but make it understandable to a tired parent.

ANTI-AI LANGUAGE RULES:
- Avoid overused filler adverbs and reactions: carefully, excitedly, happily, suddenly, softly, gently, smiled, laughed, gasped, nodded.
- Use specific actions instead: tucked, tilted, peered, balanced, whispered, offered, shuffled, patted, lifted, traced, listened, shared.
- Do not use a chain of similar adjectives such as "soft, gentle, glowing, sparkling". Choose one concrete image.
- Add one short line of natural dialogue if it helps the opening feel alive.

OPENING IDEA:
"{opening}"

OPENING RULES:
- Rewrite the opening in fresh words and continue immediately from it.
- Keep the same setting/magical idea; do not switch to a different setup.
- {opening_transition_rule}

PAGE LENGTH:
- {page_length_rule}
- 1-2 gentle paragraphs.
- {sentence_rule}.
- This limit matters because Page 1 starts narration. Move extra world-building to Page 2.

COMPANION:
- {blocks['companion_line']}

CHARACTERS:
- {blocks['character_instruction']}

OUTPUT FORMAT STRICT:
Return ONLY valid JSON:
{{"title":"Short magical title","pages":["page 1 text"]}}
"""

    def _build_first_page_fallback(self, request: GenerateStoryRequest, companion: Optional[dict]) -> Dict[str, Any]:
        """Fast polished page-1 fallback used only when Gemini is too slow.

        This must remain deterministic and instant. It protects the Page-1-first
        architecture without making fallback feel like a two-sentence placeholder.
        Do not call Gemini here and do not generate the full story here.
        """
        child = request.childName or "the child"
        language_code = (request.storyLanguageCode or "en").lower()[:2]
        expected_pages = self._intended_page_count(request)
        theme = self._localized_theme_label(request.theme, language_code) or "magic"
        theme_key = str(request.theme or "").lower().replace("-", "_").replace(" ", "_")
        moral = str(request.moral or "kindness").strip().lower()
        localized_companion = self._localized_companion(companion, language_code)

        companion_en = ""
        if localized_companion:
            companion_en = f" {localized_companion['name']} came too, carrying the clue as carefully as a biscuit on a plate."

        english_theme_setups = {
            "dragons": [
                {
                    "title": f"{child} and the Dragon Bell",
                    "page": (
                        f"Just before story time, {child} heard a polite tap on the window and found a folded note outside. "
                        f"A small dragon footprint marked the corner, and the note smelled faintly of warm toast. "
                        f"It asked for help at the mountain post office, where the bedtime bell would not ring unless everyone shared the last bundle of letters. "
                        f"{child} tucked the note close and followed a trail of harmless smoke curls toward the first step of the adventure."
                        f"{companion_en}"
                    ),
                },
                {
                    "title": f"{child} and the Teacup Dragon",
                    "page": (
                        f"A tiny puff of smoke rolled under the story-room door and stopped beside {child}'s foot. "
                        f"Inside it sat a teacup with a dragon scale for a handle and a message wrapped around the spoon. "
                        f"The message said the dragon tea party had one cake left and three hungry guests, so someone kind was needed before the kettle sang. "
                        f"{child} opened the door a little wider and stepped after the smoky trail."
                        f"{companion_en}"
                    ),
                },
            ],
            "space": [
                {
                    "title": f"{child} and the Star Signal",
                    "page": (
                        f"While {child} was choosing a bedtime story, the telescope gave three small taps against the shelf. "
                        f"Through the lens, one faraway star blinked in a pattern that looked almost like words. "
                        f"A message appeared on the glass: the sky train had too many wishes and not enough seats, and someone needed to help them share the ride. "
                        f"{child} pressed one hand to the telescope and watched a narrow path of starlight appear."
                        f"{companion_en}"
                    ),
                }
            ],
            "animals": [
                {
                    "title": f"{child} and the Pawprint Parade",
                    "page": (
                        f"Just before bedtime, {child} spotted a line of pawprints crossing the floor where no pawprints had been before. "
                        f"They led behind the story chair to a nervous rabbit holding a ribbon in both paws. "
                        f"The animals were preparing a quiet parade, but nobody could agree how to share the first drumbeat. "
                        f"{child} knelt down, listened to the rabbit's whisper, and followed the pawprints toward the garden gate."
                        f"{companion_en}"
                    ),
                }
            ],
            "princess": [
                {
                    "title": f"Princess {child} and the Shared Crown",
                    "page": (
                        f"{child} was building a pretend castle from cushions when a paper crown slid out from under the tallest tower. "
                        f"It was decorated with crayon jewels, but one side had been left blank on purpose. "
                        f"A note inside said the castle garden was waiting for a princess who could help two friends share the honour of leading the lantern walk. "
                        f"{child} picked up the crown, noticed the empty space, and stepped through an arch made from blankets."
                        f"{companion_en}"
                    ),
                },
                {
                    "title": f"{child} and the Cushion Castle Promise",
                    "page": (
                        f"Before the bedtime story began, {child} arranged cushions into a castle with a blanket bridge across the floor. "
                        f"When the bridge wrinkled, a small invitation popped out from between two cushions. "
                        f"It asked for help in the royal garden, where the last ribbon for the evening parade had to be shared fairly. "
                        f"{child} smoothed the bridge, promised to listen first, and followed the invitation through the blanket arch."
                        f"{companion_en}"
                    ),
                },
            ],
            "adventure": [
                {
                    "title": f"{child} and the Map Under the Book",
                    "page": (
                        f"When {child} lifted the bedtime book, a folded map was hiding underneath it. "
                        f"The map showed the room, the doorway, and one path that definitely had not been there before. "
                        f"At the edge, a note said someone nearby needed help sharing the last bright idea before the night settled in. "
                        f"{child} traced the path with one finger, tucked the map safely away, and took the first quiet step."
                        f"{companion_en}"
                    ),
                }
            ],
        }

        generic_english = [
            {
                "title": f"{child}'s {theme.title()} Promise",
                "page": (
                    f"Just before story time, {child} noticed something unusual beside the bedtime book. "
                    f"A folded message waited there, marked with a picture from a {theme} place and tied with one crooked thread. "
                    f"It said that two friends needed help before the evening settled, because sharing one small thing could change the whole adventure. "
                    f"{child} read the message twice, kept the crooked thread as a promise, and followed the first clue toward the doorway."
                    f"{companion_en}"
                ),
            },
            {
                "title": f"{child} and the First Promise",
                "page": (
                    f"A quiet knock came from a place where knocks did not usually come from. "
                    f"When {child} looked closer, a small sign pointed toward a {theme} problem waiting just beyond the room. "
                    f"The sign showed two hands holding the same ribbon, as if the adventure could only begin when someone chose to share. "
                    f"{child} touched the ribbon, made a careful promise to help, and stepped toward the first clue."
                    f"{companion_en}"
                ),
            },
        ]

        fallback_variants = {
            "en": english_theme_setups.get(theme_key, generic_english),
            "es": [
                {
                    "title": f"La promesa de {child}",
                    "page": (
                        f"Antes del cuento, {child} encontró un mensaje doblado junto al libro de dormir. "
                        f"Tenía un dibujo de {theme} y un hilo torcido atado en una esquina. "
                        f"El mensaje decía que dos amigos necesitaban ayuda antes de que terminara la tarde, porque compartir una cosa pequeña podía cambiar toda la aventura. "
                        f"{child} guardó el hilo como una promesa y siguió la primera pista hacia la puerta."
                    ),
                }
            ],
            "fr": [
                {
                    "title": f"La promesse de {child}",
                    "page": (
                        f"Avant l'histoire du soir, {child} trouva un message plié près du livre. "
                        f"Il portait un dessin de {theme} et un fil de travers noué dans un coin. "
                        f"Le message disait que deux amis avaient besoin d'aide avant la fin du soir, car partager une petite chose pouvait changer toute l'aventure. "
                        f"{child} garda le fil comme une promesse et suivit le premier indice vers la porte."
                    ),
                }
            ],
            "de": [
                {
                    "title": f"{child}s Versprechen",
                    "page": (
                        f"Vor der Gute-Nacht-Geschichte fand {child} eine gefaltete Nachricht neben dem Buch. "
                        f"Darauf war ein Bild von {theme} zu sehen, und an einer Ecke hing ein schiefer Faden. "
                        f"In der Nachricht stand, dass zwei Freunde Hilfe brauchten, weil Teilen ein kleines Abenteuer verändern konnte. "
                        f"{child} bewahrte den Faden wie ein Versprechen auf und folgte dem ersten Hinweis zur Tür."
                    ),
                }
            ],
            "it": [
                {
                    "title": f"La promessa di {child}",
                    "page": (
                        f"Prima della storia della sera, {child} trovò un messaggio piegato vicino al libro. "
                        f"Aveva un disegno di {theme} e un filo storto legato a un angolo. "
                        f"Il messaggio diceva che due amici avevano bisogno di aiuto, perché condividere una piccola cosa poteva cambiare tutta l'avventura. "
                        f"{child} tenne il filo come una promessa e seguì il primo indizio verso la porta."
                    ),
                }
            ],
        }

        variants = fallback_variants.get(language_code, fallback_variants["en"])
        selected = random.choice(variants)
        pages = postprocess_story_pages([selected["page"]])[:1]
        return {
            'title': selected['title'],
            'pages': pages,
            'companion': companion,
            'expected_pages': expected_pages,
            'generation_status': 'partial',
            'generation_fallback_reason': 'first_page_timeout',
        }

    def _build_remaining_pages_prompt(
        self,
        request: GenerateStoryRequest,
        companion: Optional[dict],
        title: str,
        existing_pages: list[str],
        remaining_page_count: int,
        next_page_number: int,
    ) -> str:
        """Build a compact continuation prompt for pages 2+.

        Page 1 is already generated and returned to the reader. This prompt is
        intentionally much smaller than the full Story Bible prompt so pages 2+
        can complete faster in the background without changing narration,
        chunking, subscriptions, Parent Voice, page count, or reader flow.
        """
        blocks = self._language_and_character_blocks(request, companion)
        age_rules = self._first_page_age_prompt_rules(request.age)
        humour_rule = self._age_humour_instruction(request.age)
        language_style = self._language_style_block(request.storyLanguageCode)
        existing_pages_text = "\n\n".join(
            f"Page {idx + 1}: {page}" for idx, page in enumerate(existing_pages or [])
        )
        final_page_number = next_page_number + remaining_page_count - 1

        return f"""Continue this premium bedtime story from the existing pages.

LANGUAGE:
- Write ONLY in {blocks['language_name']}.
- Do not mix languages.
{language_style}

STORY FACTS:
- Title: {title}
- Child: {request.childName}, age {request.age}
- Theme: {blocks['effective_theme']}
- Moral: {request.moral}
- Calm level: {request.calmLevel}
- Existing pages so far:
{existing_pages_text}

AGE / STYLE LOCK:
{age_rules}
{self._oxford_inspired_age_profile_block(request.age)}
- Humour guidance: {humour_rule}

CONTINUATION JOB:
- Write exactly {remaining_page_count} new pages: Page {next_page_number} through Page {final_page_number}.
- Continue naturally from the latest existing page.
- Do not recap existing pages and do not contradict them.
- Keep the same world, promise, object, helper, mood, and story direction already established.
- The story must feel like one coherent picture-book adventure, not separate scenes.
- Keep one clear goal visible from page to page.
- Each page needs one clear job: arrive, meet, notice, try, choose, solve, or settle.
- Introduce only one major new thing per page.
- Avoid adding several new names, places, objects, and rules together.

STORY QUALITY RULES:
- Show, do not explain. Avoid “learned”, “realised”, “remembered the lesson”, “explained that”, or moral lectures.
- Continue from Page 1's ordinary reason and magical trigger. Do not make the child feel randomly transported into a new story.
- Each important helper should have a simple reason for helping, worrying, hiding, making a mistake, or needing help. Show the reason through behaviour or dialogue, not explanation.
- Reveal world rules through discovery: questions, dialogue, mistakes, signs, objects behaving strangely, or characters demonstrating the rule.
- Use “explained” at most once in the whole continuation. Prefer short dialogue or visible action.
- The child must drive the outcome: notice a clue, ask a useful question, test an idea, make a mistake, adjust, and solve the key problem.
- Magical helpers may guide, worry, interrupt, or make funny mistakes, but they must not rescue the child or solve the main problem for them.
- Every page needs one clear job: arrive, meet, notice, try, choose, solve, or settle.
- Every page must include at least one of: dialogue, action, surprise, humour, emotional choice, or a visual change caused by the child.
- Include one gentle middle complication where the first idea does not fully work.
- Include 1-2 warm humour moments caused by character behaviour, misunderstanding, or a funny habit.
- If a funny trait appears early, reuse it once later as a callback or payoff.
- Give one supporting character a memorable identity: job, habit, phrase, tool, worry, or comic behaviour.
- Make the world feel alive with one small background detail, such as a side character doing a tiny job, an object misbehaving, or a custom happening nearby.
- Reuse 2-3 important details from Page 1 later with purpose.
- At least one Page 1 detail should help in the middle.
- At least one Page 1 detail should return on the final page as a visual or emotional callback.
- Give the magical place one simple rule or custom that affects both the problem and solution.
- Avoid generic object quests unless Page 1 clearly requires one.
- Avoid overusing these words: tiny, little, soft, gentle, golden, silver, shimmering, glowing, sparkling, moonlit, sleepy, carefully, excitedly, happily, smiled, laughed, gasped, nodded.
- Prefer concrete actions and memorable images over decorative adjectives.
- Use dialogue to reveal motives, worries, misunderstandings, and choices. At least two continuation pages should include natural child-friendly dialogue.
- Vary sentence openings. Do not start consecutive paragraphs with the child's name or the same character name.
- Keep the ending peaceful, specific, and emotionally earned.

PAGE LENGTH:
- Each continuation page should be 125-165 words.
- Each page should have exactly 2 gentle paragraphs.
- Each page should contain about 5-7 read-aloud sentences.
- No page should feel like a summary.
- Final page may be slightly shorter if the ending is complete and calm.

COMPANION:
- {blocks['companion_line']}

CHARACTERS:
- {blocks['character_instruction']}

OUTPUT FORMAT STRICT:
Return ONLY valid JSON:
{{{{"pages":["new page text","new page text"]}}}}
- The pages array must contain exactly {remaining_page_count} strings.
- No markdown, notes, explanations, or extra keys.
"""

    async def generate_story_first_page(self, request: GenerateStoryRequest, subscription: SubscriptionResponse) -> Dict[str, Any]:
        start_total = time.time()
        print("[PERF] ========================================")
        print(f"[PERF] generate_story_first_page START lang={request.storyLanguageCode} duration={request.durationMin}")

        companion = self._select_companion(request, subscription)
        expected_pages = self._intended_page_count(request)

        if not self.model:
            page_one = f"Once upon a time, {request.childName} discovered a quiet little path full of wonder. The stars seemed to listen as the bedtime adventure began. With a calm heart, {request.childName} stepped forward to learn something kind about {request.customTheme or self._localized_theme_label(request.theme, request.storyLanguageCode)}."
            pages = postprocess_story_pages([page_one])
            return {
                'title': f"{request.childName}'s Bedtime Adventure",
                'pages': pages[:1],
                'companion': companion,
                'expected_pages': expected_pages,
                'generation_status': 'partial',
            }

        prompt = self._build_first_page_prompt(request, companion)
        print(f"[PERF] first_page prompt chars={len(prompt)}")

        async def _generate_first_page_once(attempt_label: str) -> Dict[str, Any]:
            """Generate and validate Page 1 once.

            This helper is intentionally local to the Page-1 path. It does not
            change narration, chunking, polling, page count, storage, or the
            reader flow. It only lets us retry malformed Gemini JSON once
            before using the fast fallback.
            """
            t_attempt = time.time()
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self.model.generate_content,
                    prompt,
                    generation_config=FIRST_PAGE_GENERATION_CONFIG,
                ),
                timeout=FIRST_PAGE_SOFT_LIMIT_SECONDS,
            )
            elapsed = time.time() - t_attempt
            print(f"[PERF] first_page Gemini {attempt_label} took {elapsed:.2f}s")
            self._log_gemini_response_metadata(f"first_page_{attempt_label}", response)

            response_text = getattr(response, 'text', None)
            if not response_text or not isinstance(response_text, str):
                raise ValueError(f'Failed to generate first page on {attempt_label}')

            try:
                story_data = self._parse_json_response_with_repair(response_text, f"first_page_{attempt_label}")
            except Exception as parse_exc:
                raw_preview = str(response_text or '')[:3000]
                print(f"[DEBUG] first_page raw Gemini response parse_failed attempt={attempt_label} error={parse_exc}")
                print(f"[DEBUG] first_page raw Gemini response preview attempt={attempt_label} value={raw_preview!r}")
                raise ValueError(f'first_page_json_parse_failed:{attempt_label}: {parse_exc}') from parse_exc

            if not isinstance(story_data, dict) or 'title' not in story_data or 'pages' not in story_data:
                raise ValueError(f'Invalid first-page story format returned by AI on {attempt_label}')

            pages = postprocess_story_pages(story_data.get('pages', []))[:1]
            if not pages:
                raise ValueError(f'First-page story returned no pages on {attempt_label}')

            page_one_words = len(pages[0].split())
            page_one_chars = len(pages[0])
            print(f"[PERF] first_page_size words={page_one_words} chars={page_one_chars} source={attempt_label}")
            print(f"[PERF] first_page_ready_for_response pages=1 expected_pages={expected_pages} source={attempt_label}")
            return {
                'title': story_data['title'],
                'pages': pages,
                'companion': companion,
                'expected_pages': expected_pages,
                'generation_status': 'partial',
                'first_page_generation_source': attempt_label,
            }

        try:
            story_data = await _generate_first_page_once('gemini_primary')
            print(f"[PERF] generate_story_first_page DONE total={time.time() - start_total:.2f}s source=gemini_primary")
            print("[PERF] ========================================")
            return story_data
        except asyncio.TimeoutError:
            # Do not retry timeout. Retrying a slow first-page call would risk
            # breaking the Page-1-first speed rule. Use the instant fallback.
            print(
                f"[PERF] first_page Gemini soft limit hit after {time.time() - start_total:.2f}s; "
                "using fast fallback page 1"
            )
            fallback = self._build_first_page_fallback(request, companion)
            fallback['generation_fallback_reason'] = 'first_page_timeout'
            fallback['first_page_generation_source'] = 'fallback_timeout'
            fallback_page = (fallback.get('pages') or [''])[0]
            print(f"[PERF] first_page_size fallback words={len(fallback_page.split())} chars={len(fallback_page)} source=fallback_timeout")
            print(f"[PERF] generate_story_first_page DONE fallback total={time.time() - start_total:.2f}s reason=timeout")
            print("[PERF] ========================================")
            return fallback
        except Exception as first_exc:
            # Retry only malformed/invalid Gemini output once. This reduces
            # repeated deterministic fallback stories while preserving speed for
            # the normal successful path and avoiding retries on timeouts.
            print(f"[PERF] first_page primary failed; retrying once before fallback: {first_exc}")
            try:
                story_data = await _generate_first_page_once('gemini_retry')
                print(f"[PERF] generate_story_first_page DONE total={time.time() - start_total:.2f}s source=gemini_retry")
                print("[PERF] ========================================")
                return story_data
            except asyncio.TimeoutError:
                print(
                    f"[PERF] first_page retry soft limit hit after {time.time() - start_total:.2f}s; "
                    "using fast fallback page 1"
                )
                fallback_reason = 'first_page_retry_timeout'
                fallback_source = 'fallback_retry_timeout'
                fallback_exception = 'retry_timeout'
            except Exception as retry_exc:
                print(f"[PERF] first_page retry failed, using deterministic page 1 fallback: {retry_exc}")
                fallback_reason = 'first_page_retry_exception'
                fallback_source = 'fallback_retry_exception'
                fallback_exception = str(retry_exc)

            fallback = self._build_first_page_fallback(request, companion)
            fallback['generation_fallback_reason'] = fallback_reason
            fallback['first_page_generation_source'] = fallback_source
            fallback['first_page_fallback_detail'] = fallback_exception[:300]
            fallback_page = (fallback.get('pages') or [''])[0]
            print(f"[PERF] first_page_size fallback words={len(fallback_page.split())} chars={len(fallback_page)} source={fallback_source}")
            print(f"[PERF] generate_story_first_page DONE fallback total={time.time() - start_total:.2f}s reason={fallback_reason}")
            print("[PERF] ========================================")
            return fallback

    async def complete_story_background(
        self,
        request: GenerateStoryRequest,
        user_id: str,
        story_id: str,
        title: str,
        current_pages: list[str],
        companion: Optional[dict],
        expected_pages: int,
    ) -> None:
        start_total = time.time()
        print(f"[PERF] complete_story_background START story_id={story_id}")
        try:
            if not self.model:
                remaining = [
                    f"On the next part of the path, {request.childName} found a small kindness waiting to be shared.",
                    f"The quiet adventure grew softer and brighter as {request.childName} remembered what mattered most.",
                    f"At last, the moon smiled gently, and {request.childName} felt safe, loved, and ready for sleep.",
                ]
                while len(current_pages) + len(remaining) < expected_pages:
                    remaining.append(f"A peaceful little moment helped {request.childName} feel even calmer.")
            else:
                remaining_count = max(expected_pages - len(current_pages), 0)
                if remaining_count <= 0:
                    self.story_repo.update(story_id, user_id, {
                        'generation_status': 'complete',
                        'expected_pages': expected_pages,
                        'generation_error': None,
                    })
                    return

                remaining = []
                working_pages = postprocess_story_pages(current_pages)[:expected_pages]

                while len(working_pages) < expected_pages:
                    batch_count = min(BACKGROUND_PAGE_BATCH_SIZE, expected_pages - len(working_pages))
                    batch_pages = await self._generate_remaining_pages_with_recovery(
                        request=request,
                        companion=companion,
                        title=title,
                        working_pages=working_pages,
                        preferred_batch_count=batch_count,
                    )

                    working_pages = postprocess_story_pages([*working_pages, *batch_pages])[:expected_pages]
                    remaining.extend(batch_pages)

                    # Publish partial pages immediately. Reader polling can then
                    # advance to pages 2+ without waiting for the full story.
                    if len(working_pages) < expected_pages:
                        partial_text = '\n\n'.join(working_pages)
                        t_partial_update = time.time()
                        print(
                            f"[PERF] story_update_partial START story_id={story_id} "
                            f"pages={len(working_pages)}/{expected_pages}"
                        )
                        self.story_repo.update(story_id, user_id, {
                            'pages': working_pages,
                            'full_text': partial_text,
                            'generation_status': 'partial',
                            'expected_pages': expected_pages,
                            'generation_error': None,
                        })
                        print(
                            f"[PERF] story_update_partial DONE story_id={story_id} "
                            f"total={time.time() - t_partial_update:.2f}s"
                        )

            all_pages = postprocess_story_pages([*current_pages, *remaining])[:expected_pages]
            if len(all_pages) < expected_pages:
                raise ValueError(f'Remaining generation produced only {len(all_pages)} of {expected_pages} pages')

            full_text = '\n\n'.join(all_pages)
            update_payload = {
                'pages': all_pages,
                'full_text': full_text,
                'generation_status': 'complete',
                'expected_pages': expected_pages,
                'generation_error': None,
            }

            # Make pages 2+ available immediately. Metadata is useful for the
            # library, but it must never delay the reader/polling flow.
            t_update = time.time()
            print(f"[PERF] story_update_complete START story_id={story_id}")
            self.story_repo.update(story_id, user_id, update_payload)
            print(f"[PERF] story_update_complete DONE story_id={story_id} total={time.time() - t_update:.2f}s")

            try:
                t_metadata = time.time()
                print(f"[PERF] metadata_extract START story_id={story_id}")
                metadata = await self.extract_metadata(title, full_text)
                print(f"[PERF] metadata_extract DONE story_id={story_id} total={time.time() - t_metadata:.2f}s")
                t_metadata_update = time.time()
                print(f"[PERF] metadata_update START story_id={story_id}")
                self.story_repo.update(story_id, user_id, {
                    'story_summary': metadata.get('summary', ''),
                    'characters': metadata.get('characters', []),
                    'setting': metadata.get('setting', ''),
                })
                print(f"[PERF] metadata_update DONE story_id={story_id} total={time.time() - t_metadata_update:.2f}s")
            except Exception as metadata_exc:
                print(f"[PERF] metadata_extract skipped story_id={story_id}: {metadata_exc}")

            print(f"[PERF] complete_story_background DONE story_id={story_id} pages={len(all_pages)} total={time.time() - start_total:.2f}s")
        except Exception as exc:
            print(f"[PERF] complete_story_background FAILED story_id={story_id}: {exc}")
            try:
                self.story_repo.update(story_id, user_id, {
                    'generation_status': 'failed',
                    'expected_pages': expected_pages,
                    'generation_error': str(exc)[:500],
                })
            except Exception as update_exc:
                print(f"[PERF] failed to mark story generation failed story_id={story_id}: {update_exc}")

    async def generate_story(self, request: GenerateStoryRequest, subscription: SubscriptionResponse) -> Dict[str, Any]:
        start_total = time.time()
        print("[PERF] ========================================")
        print(f"[PERF] generate_story START lang={request.storyLanguageCode} duration={request.durationMin}")

        companion = self._select_companion(request, subscription)
        print(f"[PERF] companion selected in {time.time() - start_total:.2f}s has_companion={bool(companion)}")

        if not self.model:
            pages = [
                f"Once upon a time, {request.childName} discovered a quiet little path full of wonder.",
                f"The path led to a gentle adventure about {request.customTheme or self._localized_theme_label(request.theme, request.storyLanguageCode)}, where kindness mattered most.",
                f"Soon, everything grew peaceful again, and {request.childName} felt calm enough for sleep.",
            ]
            print(f"[PERF] fallback story returned in {time.time() - start_total:.2f}s")
            print("[PERF] ========================================")
            return {'title': f"{request.childName}'s Bedtime Adventure", 'pages': postprocess_story_pages(pages), 'companion': companion}

        t_prompt = time.time()
        prompt = self._build_prompt(request, companion)
        print(f"[PERF] prompt built in {time.time() - t_prompt:.2f}s chars={len(prompt)}")

        t_gemini = time.time()
        response = self.model.generate_content(
            prompt,
            generation_config=FULL_STORY_GENERATION_CONFIG,
        )
        print(f"[PERF] Gemini generate_content took {time.time() - t_gemini:.2f}s")
        self._log_gemini_response_metadata("full_story", response)

        response_text = getattr(response, 'text', None)
        if not response_text or not isinstance(response_text, str):
            print(f"[PERF] generate_story FAILED no response text total={time.time() - start_total:.2f}s")
            print("[PERF] ========================================")
            raise HTTPException(status_code=500, detail='Failed to generate story')

        t_clean = time.time()
        cleaned = response_text.strip()
        if cleaned.startswith('```json'):
            cleaned = cleaned[7:]
        if cleaned.startswith('```'):
            cleaned = cleaned[3:]
        if cleaned.endswith('```'):
            cleaned = cleaned[:-3]
        print(f"[PERF] cleaning took {time.time() - t_clean:.2f}s response_chars={len(response_text)}")

        t_parse = time.time()
        story_data = self._parse_json_response_with_repair(cleaned.strip(), "full_story")
        print(f"[PERF] JSON parse took {time.time() - t_parse:.2f}s")

        if not isinstance(story_data, dict) or 'title' not in story_data or 'pages' not in story_data:
            print(f"[PERF] generate_story FAILED invalid format total={time.time() - start_total:.2f}s")
            print("[PERF] ========================================")
            raise HTTPException(status_code=500, detail='Invalid story format returned by AI')

        t_post = time.time()
        pages = postprocess_story_pages(story_data.get('pages', []))
        print(f"[PERF] postprocess took {time.time() - t_post:.2f}s pages_before_trim={len(pages)}")

        # Hard guard for production performance: Gemini may occasionally exceed
        # the requested page count. Trim to the intended count so narration cost,
        # timing, and reader sync remain predictable.
        intended_page_count = 7
        story_data['pages'] = pages[:intended_page_count]
        story_data['companion'] = companion

        total_words = sum(len(str(page).split()) for page in story_data['pages'])
        print(
            f"[PERF] generate_story DONE total={time.time() - start_total:.2f}s "
            f"lang={request.storyLanguageCode} pages={len(story_data['pages'])} words={total_words}"
        )
        print("[PERF] ========================================")
        return story_data

    async def extract_metadata(self, title: str, full_text: str) -> Dict[str, Any]:
        start_total = time.time()
        if not self.model:
            print(f"[PERF] extract_metadata skipped no_model total={time.time() - start_total:.2f}s")
            return {'summary': '', 'characters': [], 'setting': ''}
        prompt = (
            'Analyze this bedtime story and return only valid JSON. '
            'Schema: {"summary":"...","characters":[{"name":"...","description":"...","role":"..."}],"setting":"..."}\n'
            f'Title: {title}\nStory:\n{full_text}'
        )
        try:
            t_gemini = time.time()
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=METADATA_GENERATION_CONFIG,
            )
            print(f"[PERF] extract_metadata Gemini took {time.time() - t_gemini:.2f}s")
            self._log_gemini_response_metadata("metadata", response)
            text = getattr(response, 'text', '')
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1:
                print(f"[PERF] extract_metadata invalid_json total={time.time() - start_total:.2f}s")
                return {'summary': '', 'characters': [], 'setting': ''}
            result = json.loads(text[start:end + 1])
            print(f"[PERF] extract_metadata DONE total={time.time() - start_total:.2f}s")
            return result
        except Exception as exc:
            print(f"[PERF] extract_metadata FAILED total={time.time() - start_total:.2f}s error={exc}")
            return {'summary': '', 'characters': [], 'setting': ''}

    def validate_story_limits(self, user_id: str, subscription: SubscriptionResponse) -> None:
        tier = SUBSCRIPTION_TIERS['premium' if subscription.is_premium else 'free']
        week_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        stories_this_week = self.story_repo.count_since(user_id, week_ago)
        if tier['weekly_story_limit'] is not None and stories_this_week >= tier['weekly_story_limit']:
            raise HTTPException(status_code=403, detail={'error': 'story_limit_reached', 'message': "You've created 2 free stories this week. Upgrade to create unlimited bedtime stories.", 'upgrade_required': True})
        stories_saved = self.story_repo.count_all(user_id)
        if tier['max_saved_stories'] is not None and stories_saved >= tier['max_saved_stories']:
            raise HTTPException(status_code=403, detail={'error': 'storage_limit', 'message': "You've reached the maximum number of saved stories.", 'upgrade_required': True})

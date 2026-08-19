from __future__ import annotations

import asyncio
import json
import os
import random
import re
import time
from google import genai
from google.genai import types

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from fastapi import HTTPException

# Load local .env before reading settings. This is safe on Render too:
# Render environment variables already exist, and load_dotenv will not override them.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # python-dotenv is optional in production. If it is not installed, the
    # direct os.getenv fallback in StoryService.__init__ still works for
    # already-exported environment variables.
    pass

from app.core.config import settings
from app.domain.constants import STORY_COMPANIONS, SUBSCRIPTION_TIERS, SUPPORTED_LANGUAGES
from app.models.story import GenerateStoryRequest
from app.models.subscription import SubscriptionResponse
from app.repositories.story_repository import StoryRepository
from app.repositories.story_world_repository import StoryWorldRepository
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
FIRST_PAGE_SOFT_LIMIT_SECONDS = 18

# Background continuation is generated one page at a time so Reader polling
# receives each next page as soon as it is ready. This avoids the page-4
# stall where a later multi-page batch can fail and leave the story paused
# after pages 1-3, while preserving Page-1-first playback.
BACKGROUND_PAGE_BATCH_SIZE = 1

# Background continuation must not hang forever on one provider request.
# Page 1 remains on the existing fast path; this applies only to Pages 2+.
BACKGROUND_PAGE_TIMEOUT_SECONDS = 12
BACKGROUND_PAGE_MAX_ATTEMPTS = 2

# If all immediate continuation attempts for the next page fail (for example,
# Gemini returns HTTP 200 with empty response text), do not permanently strand
# an otherwise playable partial story. Retry the SAME next page after a short
# provider-cooldown delay before allowing the background task to pause.
#
# This applies only to Pages 2+ and does not alter the Page-1-first path.
BACKGROUND_CONTINUATION_RECOVERY_ATTEMPTS = 2
BACKGROUND_CONTINUATION_RECOVERY_DELAY_SECONDS = 1.5

# Canon-only Story Worlds release. Folk Adventures stay implemented but are
# disabled by default until their rules/content layer is production-ready.
ENABLE_FOLK_ADVENTURES = os.getenv("ENABLE_FOLK_ADVENTURES", "false").strip().lower() in {"1", "true", "yes", "on"}


class StoryService:
    def __init__(self, story_repo: StoryRepository, story_world_repo: Optional[StoryWorldRepository] = None):
        self.story_repo = story_repo
        self.story_world_repo = story_world_repo

        # Prefer config.py settings, but fall back directly to environment
        # variables so local .env loading issues do not silently disable Gemini.
        # Never print the API key itself.
        gemini_api_key = (getattr(settings, "gemini_api_key", "") or os.getenv("GEMINI_API_KEY", "")).strip()
        gemini_model = (getattr(settings, "gemini_model", "") or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()

        if gemini_api_key:
            self.client = genai.Client(api_key=gemini_api_key)
            self.model_name = gemini_model
            # Backward-compatible truthiness guard used throughout this service.
            # Generation itself runs through self.client.models.
            self.model = self.model_name
        else:
            self.client = None
            self.model_name = None
            self.model = None

        print(
            "[CONFIG] Gemini status: "
            f"api_key_loaded={bool(gemini_api_key)} "
            f"model={self.model_name or 'not_configured'} "
            f"settings_key_loaded={bool(getattr(settings, 'gemini_api_key', ''))} "
            f"env_key_loaded={bool(os.getenv('GEMINI_API_KEY'))}"
        )
        print("[BUILD] StoryService canon_release_hardened continuation_recovery=20260814 multilingual_canon_validation=20260814 multilingual_canon_scene_fallback=20260814 multilingual_final_page_validation=20260814 canon_instruction_leak_guard=20260816 bedtime_quality_restore=20260816 canon_event_budget=20260816 canon_oxford_storytelling=20260816 canon_age_safety_law=20260816 natural_name_pronouns=20260816 page_boundary_dedupe=20260817 bedtime_elite_quality=20260819 plain_prose_guard=20260819 bedtime_author_voice_98=20260819 hidden_child_age=20260819")

    def _normalise_story_world_mode(self, request: GenerateStoryRequest) -> str:
        raw = str(getattr(request, 'storyWorldMode', '') or '').strip().lower()
        aliases = {
            'traditional_folk_tale': 'canon',
            'original_folk_story': 'canon',
            'canon': 'canon',
            'adapted_folk_tale': 'folk_adventure',
            'pillowtales_folk_adventure': 'folk_adventure',
            'folk_adventure': 'folk_adventure',
            'adapted': 'folk_adventure',
        }
        return aliases.get(raw, raw)

    def _is_canon_request(self, request: GenerateStoryRequest) -> bool:
        return self._normalise_story_world_mode(request) == 'canon'

    def _is_folk_adventure_request(self, request: GenerateStoryRequest) -> bool:
        return self._normalise_story_world_mode(request) == 'folk_adventure'


    def _moral_requested(self, request: GenerateStoryRequest) -> bool:
        raw = str(getattr(request, 'moral', '') or '').strip().lower()
        return raw not in {'', 'none', 'no moral', 'no_moral', 'null', 'undefined'}

    def _select_living_world_episode_seed(self, request: GenerateStoryRequest, context: Optional[dict] = None) -> Optional[dict]:
        """Select and cache one continuity seed locally for the full request.

        This adds no model call and therefore preserves the Page-1-first path.
        """
        context = context or self._resolve_story_world_context(request)
        if not context or context.get('mode') != 'folk_adventure':
            return None
        cached = context.get('living_world_episode_seed')
        if isinstance(cached, dict):
            return cached
        continuity = context.get('living_world_continuity') or {}
        content = continuity.get('content') if isinstance(continuity, dict) else {}
        content = content if isinstance(content, dict) else {}
        seeds = [seed for seed in (content.get('story_seeds') or []) if isinstance(seed, dict)]
        selected = random.choice(seeds) if seeds else None
        context['living_world_episode_seed'] = selected
        return selected

    def _living_world_prompt_payload(self, request: GenerateStoryRequest) -> dict:
        context = self._resolve_story_world_context(request) or {}
        source_canon = context.get('source_canon') or {}
        continuity = context.get('living_world_continuity') or {}
        episode_seed = self._select_living_world_episode_seed(request, context)
        return {
            'source_canon_version': source_canon.get('version'),
            'source_canon': source_canon.get('content') or {},
            'continuity_version': continuity.get('version'),
            'living_world_continuity': continuity.get('content') or {},
            'selected_episode_seed': episode_seed or {},
        }

    def _resolve_story_world_context(self, request: GenerateStoryRequest) -> Optional[dict]:
        """Resolve and cache one stable Story World context per request."""
        slug = str(getattr(request, 'storyWorldSlug', '') or '').strip().lower()
        mode = self._normalise_story_world_mode(request)
        if not slug or not mode:
            return None

        if mode == 'folk_adventure' and not ENABLE_FOLK_ADVENTURES:
            raise HTTPException(
                status_code=404,
                detail='PillowTales Folk Adventures are not enabled in this release',
            )

        cached = getattr(request, '_pillowtales_story_world_context', None)
        if isinstance(cached, dict):
            cached_world = cached.get('world') or {}
            if str(cached_world.get('slug') or '').strip().lower() == slug and cached.get('mode') == mode:
                return cached

        if not self.story_world_repo:
            raise HTTPException(status_code=500, detail='Story World repository is not configured')

        context = self.story_world_repo.get_generation_context(
            slug=slug,
            language_code=request.storyLanguageCode,
            age=self._safe_child_age(request.age),
            mode=mode,
        )
        if not context or not context.get('prompt_pack'):
            raise HTTPException(status_code=404, detail='Published Story World prompt pack not found')

        canon_stories = context.get('canon_stories') or []
        requested_anchor_slug = str(getattr(request, 'storyWorldAnchorSlug', '') or '').strip().lower()

        if requested_anchor_slug:
            selected_anchor = next(
                (row for row in canon_stories if str(row.get('slug') or '').strip().lower() == requested_anchor_slug),
                None,
            )
            if not selected_anchor:
                raise HTTPException(
                    status_code=409,
                    detail='Selected folklore story is not available for this child age or Story World',
                )
            if mode == 'folk_adventure':
                if not bool(selected_anchor.get('living_world_expansion_allowed')):
                    raise HTTPException(
                        status_code=409,
                        detail='Selected folklore story is not available for PillowTales Folk Adventure',
                    )
                generation_rules = selected_anchor.get('generation_rules') or {}
                folk_rules = generation_rules.get('folk_adventure') if isinstance(generation_rules, dict) else None
                if isinstance(folk_rules, dict) and folk_rules.get('allowed') is False:
                    raise HTTPException(
                        status_code=409,
                        detail='Selected folklore story is not available for PillowTales Folk Adventure',
                    )
            context['anchor'] = selected_anchor
        elif canon_stories:
            eligible_rows = canon_stories
            if mode == 'folk_adventure':
                eligible_rows = []
                for row in canon_stories:
                    if not bool(row.get('living_world_expansion_allowed')):
                        continue
                    generation_rules = row.get('generation_rules') or {}
                    folk_rules = generation_rules.get('folk_adventure') if isinstance(generation_rules, dict) else None
                    if isinstance(folk_rules, dict) and folk_rules.get('allowed') is False:
                        continue
                    eligible_rows.append(row)
            if eligible_rows:
                context['anchor'] = random.choice(eligible_rows)
            elif mode == 'canon':
                raise HTTPException(status_code=409, detail='No published original folk story is available for this age')
            else:
                raise HTTPException(status_code=409, detail='No published Folk Adventure source is available for this age')
        elif mode == 'canon':
            raise HTTPException(status_code=409, detail='No published original folk story is available for this age')
        else:
            raise HTTPException(status_code=409, detail='No published Folk Adventure source is available for this age')

        context['mode'] = mode
        if mode == 'folk_adventure':
            seed = self._select_living_world_episode_seed(request, context)
            print(
                "[STORY_WORLD_CONTEXT] "
                f"mode={mode} slug={slug} "
                f"source_canon_loaded={bool(context.get('source_canon'))} "
                f"continuity_loaded={bool(context.get('living_world_continuity'))} "
                f"episode_seed={str((seed or {}).get('title') or '')!r}"
            )
        try:
            setattr(request, '_pillowtales_story_world_context', context)
        except Exception:
            pass
        return context

    @staticmethod
    def _canon_value(anchor: dict, *keys: str, default: Any = None) -> Any:
        """Read Canon values from the actual Story World record shape.

        Canon story columns live at the top level, while detailed enforcement
        data is stored inside generation_rules. Older records may also carry a
        content object, so keep that as a backward-compatible fallback.
        """
        containers = [anchor]
        generation_rules = anchor.get('generation_rules')
        if isinstance(generation_rules, dict):
            containers.append(generation_rules)
        content = anchor.get('content')
        if isinstance(content, dict):
            containers.append(content)

        for container in containers:
            for key in keys:
                value = container.get(key)
                if value not in (None, '', [], {}):
                    return value
        return default

    @staticmethod
    def _canon_display_title(anchor: dict) -> str:
        """Return the published title for the requested generation language.

        The repository attaches the selected catalogue translation as
        ``_story_translation``. Only the display title is localised here;
        authoritative Canon characters, events, locations and generation rules
        remain on the base Canon record.
        """
        translation = anchor.get('_story_translation')
        if isinstance(translation, dict):
            translated_title = str(translation.get('title') or '').strip()
            if translated_title:
                return translated_title

        return str(
            StoryService._canon_value(
                anchor,
                'title',
                'official_title',
                'canonical_title',
                default='Original Folk Story',
            )
        )

    def _canon_contract(self, request: GenerateStoryRequest) -> dict:
        context = self._resolve_story_world_context(request)
        if not context or context.get('mode') != 'canon':
            return {}
        anchor = context.get('anchor') or {}
        title = self._canon_display_title(anchor)
        return {
            'title': str(title),
            'overview': self._canon_value(anchor, 'overview', 'synopsis', 'summary', 'canonical_summary', default=''),
            'historical_context': self._canon_value(anchor, 'historical_context', 'source_context', 'tradition_context', default=''),
            'characters': self._canon_value(anchor, 'main_characters', 'characters', 'required_characters', 'principal_characters', default=[]),
            'locations': self._canon_value(anchor, 'locations', 'required_locations', default=[]),
            'required_scenes': self._canon_value(anchor, 'required_scenes', 'scenes', 'scene_sequence', default=[]),
            'required_events': self._canon_value(anchor, 'required_events', 'events', 'event_sequence', default=[]),
            'required_event_order': self._canon_value(anchor, 'required_event_order', 'event_order', default=[]),
            'forbidden_additions': self._canon_value(anchor, 'forbidden_additions', 'forbidden_changes', 'prohibited_inventions', default=[]),
            'allowed_embellishments': self._canon_value(anchor, 'allowed_embellishments', 'permitted_adaptations', default=[]),
            'child_insertion_rules': self._canon_value(anchor, 'child_insertion_rules', 'child_role', default=[]),
            'ending_rules': self._canon_value(anchor, 'required_ending', 'ending_rules', 'canonical_ending', 'ending', default=''),
            'validation_rules': self._canon_value(anchor, 'validation_rules', 'canon_validation', default=[]),
            'bedtime_adaptation': self._canon_value(anchor, 'bedtime_adaptation', default=''),
        }

    def _folk_adventure_contract(self, request: GenerateStoryRequest) -> dict:
        """Return the selected folklore source as an expansion contract.

        Folk Adventure is creative, but it must exist because the selected
        folklore source exists. The same request-level Story World context is
        reused for Page 1, Pages 2-7, retries and metadata persistence.
        """
        context = self._resolve_story_world_context(request)
        if not context or context.get('mode') != 'folk_adventure':
            return {}
        anchor = context.get('anchor') or {}
        generation_rules = anchor.get('generation_rules') or {}
        folk_rules = generation_rules.get('folk_adventure') or {}

        return {
            'source_title': str(anchor.get('title') or ''),
            'source_slug': str(anchor.get('slug') or ''),
            'summary': anchor.get('summary') or '',
            'characters': anchor.get('main_characters') or [],
            'locations': anchor.get('locations') or [],
            'creatures': anchor.get('creatures') or [],
            'core_values': anchor.get('core_values') or [],
            'protected_facts': folk_rules.get('protected_facts') or [],
            'valid_timeframes': folk_rules.get('valid_timeframes') or [],
            'valid_entry_points': folk_rules.get('valid_entry_points') or [],
            'expandable_consequences': folk_rules.get('expandable_consequences') or [],
            'forbidden_contradictions': folk_rules.get('forbidden_contradictions') or [],
            'allowed': folk_rules.get('allowed', True),
        }

    def _folk_adventure_contract_block(self, request: GenerateStoryRequest) -> str:
        contract = self._folk_adventure_contract(request)
        if not contract:
            return ''
        payload = json.dumps(contract, ensure_ascii=False, separators=(',', ':'))
        return f"""FOLK ADVENTURE SOURCE CONTRACT — AUTHORITATIVE:
{payload}

SOURCE DEPENDENCY RULES:
- This is a new Living World episode, not a retelling and not a repair of the legend.
- Preserve all protected facts and forbidden contradictions.
- The selected source is historical foundation and identity; it is not a compulsory plot template.
- Legendary characters may lead the episode when compatible with Source Canon.
- The listening child is not a character in Living World episodes. The child's name may appear only in the brief external Page 1 bedtime invitation; once the episode begins, do not insert, address, describe, or refer to the listener inside the plot.
- Do not invent a missing, forgotten, corrected, repaired, recovered, secret, or alternative part of the recorded legend.
- Never make any new character responsible for completing or correcting canon.
- Protected names retain exact spelling, accents, spacing and capitalisation.
- Pronunciation guidance changes narration only, never written spelling.
"""

    def _canon_contract_block(self, request: GenerateStoryRequest) -> str:
        contract = self._canon_contract(request)
        if not contract:
            return ''
        payload = json.dumps(contract, ensure_ascii=False, separators=(',', ':'))
        return f"""CANON SOURCE OF TRUTH — AUTHORITATIVE:
{payload}

CANON AUTHORITY RULES:
- Retell this record. Do not create a new plot inspired by it.
- Preserve the exact canonical title, characters, locations, events, sequence and ending.
- Any Canon character or location marked protected=true is immutable display text. Use its stored spelling, accents, spacing and capitalisation exactly every time it appears.
- Never anglicise, translate, simplify, modernise, respell, de-accent or change the capitalisation of a protected Canon name.
- A shortened reference is allowed only when it is an exact component of the protected stored name (for example, Fionn from Fionn mac Cumhaill). Never substitute a different variant such as Finn.
- Pronunciation guidance affects narration only; it must never change the written story spelling.
- Do not invent objects, quests, morals, motivations, titles, villains, helpers, solutions or endings.
- Ignore the parent's theme and moral as plot instructions.
- The listening child must remain completely outside the Canon story.
- Do not mention the child's name anywhere in the generated story prose.
- Do not address, describe or refer to the listening child in the opening, body, ending, epilogue, dialogue, framing device or bedtime outro.
- Do not invent a parent, grown-up, narrator or storyteller speaking to the listening child.
- Only canonical characters may appear or participate in the story.
- Personalisation may affect age-appropriate vocabulary, sentence complexity, pacing, intensity, length and bedtime tone only. It must not insert the listening child into the prose.
- These rules override child_insertion_rules for Original Folk Stories. Canon stories never include the listening child.
- Bedtime-safe connective details may clarify a canonical departure without changing it: when someone leaves home, family, guardians or companions for an extended journey, do not accidentally imply an unexplained disappearance if the source permits a brief acknowledgement, farewell, permission, witnessed departure or equivalent reassurance.
- Such connective detail must never change the canonical decision, event, relationship, consequence or ending.
- Improve only age clarity, dialogue, pacing, warmth and bedtime readability.
- When a general creative rule conflicts with canon, canon wins.
"""

    def _story_world_prompt_block(self, request: GenerateStoryRequest) -> str:
        context = self._resolve_story_world_context(request)
        if not context:
            return ''

        world = context['world']
        prompt_pack = context.get('prompt_pack') or {}
        story_dna = context.get('story_dna') or {}
        editorial = context.get('editorial_bible') or {}
        anchor = context.get('anchor') or {}
        mode = context.get('mode')

        pack_content = json.dumps(prompt_pack.get('content') or {}, ensure_ascii=False, separators=(',', ':'))
        dna_content = json.dumps(story_dna.get('content') or {}, ensure_ascii=False, separators=(',', ':'))
        editorial_content = json.dumps(editorial.get('content') or {}, ensure_ascii=False, separators=(',', ':'))
        anchor_content = json.dumps(anchor, ensure_ascii=False, separators=(',', ':'))

        if mode == 'canon':
            return f"""STORY WORLD CONTEXT — CANON ENGINE:
- Story World slug: {world.get('slug')}
- Story World category: {world.get('category')}
- Prompt pack version: {prompt_pack.get('version')}

{self._canon_contract_block(request)}

STORY WORLD DNA — STYLE ONLY:
{dna_content}

EDITORIAL AND CULTURAL BOUNDARIES:
{editorial_content}

CANON LAYERING RULE:
- Canon controls facts, order and ending.
- Story DNA controls voice and atmosphere only.
- Parent theme, parent moral, random opening seeds and invented plot engines are disabled.
"""

        living_payload = self._living_world_prompt_payload(request)
        living_content = living_payload.get('living_world_continuity') or {}
        anti_generic = living_content.get('anti_generic_failures') or []
        quality_contract = living_content.get('episode_quality_contract') or []
        transplant_test = living_content.get('transplant_test') or ''
        protagonist_rule = (
            "Use an established legendary, recurring, or continuity-approved world character as protagonist. "
            "The listening child remains entirely outside the episode."
        )
        return f"""STORY WORLD CONTEXT — LIVING WORLD ENGINE:
- Story World slug: {world.get('slug')}
- Story World category: {world.get('category')}
- Story World type: {world.get('world_type')}
- Prompt pack version: {prompt_pack.get('version')}

MODE: PILLOWTALES LIVING WORLD
- Create a new episode inside this continuing Story World.
- {protagonist_rule}
- The only listener-facing material allowed is a brief external Page 1 bedtime invitation. The actual episode must begin inside the selected Story World, not in the listener's bedroom, imagination, dream, beach, home, or ordinary life.
- Do not frame the episode as a symbolic memory of the country.
- No parent-selected moral or parent-selected theme drives this mode.
- Begin with an active world-specific situation involving a place, character, institution, creature, conflict, discovery or event from continuity.
- The selected folklore anchor protects identity and supplies foundations; it does not force the child into the plot.

{self._folk_adventure_contract_block(request)}

PROTECTED SOURCE CANON — WORLD LEVEL:
{json.dumps(living_payload.get('source_canon') or {}, ensure_ascii=False, separators=(',', ':'))}

PILLOWTALES LIVING WORLD CONTINUITY — AUTHORITATIVE FOR NEW EPISODES:
{json.dumps(living_content, ensure_ascii=False, separators=(',', ':'))}

SELECTED EPISODE SEED — USE AS THE PRIMARY PREMISE:
{json.dumps(living_payload.get('selected_episode_seed') or {}, ensure_ascii=False, separators=(',', ':'))}

SELECTED FOLKLORE SOURCE RECORD — PROTECTED FOUNDATION:
{anchor_content}

COMPILED STORY WORLD PROMPT PACK:
{pack_content}

STORY WORLD DNA:
{dna_content}

EDITORIAL AND CULTURAL BOUNDARIES:
{editorial_content}

LIVING WORLD HARD RULES:
- The listening child may appear only in the brief external Page 1 bedtime invitation. The child MUST NOT appear in the episode title, plot narration, dialogue, dream, memory, cameo, witness role, helper role, solution, climax, or ending.
- The protagonist must be an established legendary, recurring, or continuity-approved world character.
- The story must materially use this world's continuity, geography, characters, powers or institutions.
- The selected episode seed controls the central premise unless it conflicts with Source Canon.
- Do not use whispers, humming shells, shimmering fragments, forgotten memories, lost wishes, glowing clues, arbitrary portals or generic magical objects as the plot engine.
- Do not turn Ireland or another world into a vague memory, feeling or symbol.
- Do not solve the plot with gentleness, kindness or another moral unless the events naturally require that choice; no moral is requested here.
- Magic may create possibilities or complications but must not automatically solve the problem.
- WORLD ISOLATION: use only the selected Story World's canon, continuity, characters, places, powers and institutions. Never import another Story World's identity.
- LANGUAGE ISOLATION: all ordinary narration, dialogue, descriptions and non-protected invented labels must be natural {SUPPORTED_LANGUAGES.get(request.storyLanguageCode, 'English')}. Do not expose English database prose verbatim merely because source data is stored in English.
- Protected cultural names and protected proper nouns keep their exact stored spelling. Descriptive or PillowTales-invented place labels that are not protected proper names should be translated naturally and consistently into the requested story language.
- Transplant Test: {transplant_test}
- Explicit anti-generic failures: {json.dumps(anti_generic, ensure_ascii=False)}
- Episode quality contract: {json.dumps(quality_contract, ensure_ascii=False)}
"""

    def get_story_world_generation_metadata(self, request: GenerateStoryRequest) -> dict:
        context = self._resolve_story_world_context(request)
        if not context:
            return {}
        prompt_pack = context.get('prompt_pack') or {}
        anchor = context.get('anchor') or {}
        return {
            'story_world_slug': context['world'].get('slug'),
            'story_world_mode': context.get('mode'),
            'story_world_anchor_slug': anchor.get('slug'),
            'story_world_anchor_title': (
                self._canon_display_title(anchor)
                if context.get('mode') == 'canon'
                else anchor.get('title')
            ),
            'story_world_prompt_pack_version': prompt_pack.get('version'),
        }

    def _json_generation_config(
        self,
        response_schema: Optional[dict] = None,
        max_output_tokens: Optional[int] = None,
    ) -> types.GenerateContentConfig:
        """Create a Gemini JSON-mode config for the google.genai SDK.

        response_mime_type keeps output biased toward JSON. response_json_schema
        supplies the exact JSON schema used by the story and metadata calls.
        max_output_tokens is set per call so the speed-critical Page 1 path has
        enough room without changing the Page-1-first reader flow.
        """
        kwargs: Dict[str, Any] = {
            "response_mime_type": "application/json",
            # PillowTales needs fast, complete JSON rather than long internal reasoning.
            # Gemini 2.5 Flash otherwise spends much of the output budget on thinking,
            # which can truncate Page 1 and delay every continuation page.
            "thinking_config": types.ThinkingConfig(thinking_budget=0),
        }
        if max_output_tokens:
            kwargs["max_output_tokens"] = max_output_tokens
        if response_schema:
            kwargs["response_json_schema"] = response_schema
        return types.GenerateContentConfig(**kwargs)

    def _story_response_schema(self, page_count: int, include_title: bool = False) -> dict:
        properties: Dict[str, Any] = {
            "pages": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": page_count,
                "maxItems": page_count,
            }
        }
        required = ["pages"]
        if include_title:
            properties = {"title": {"type": "string"}, **properties}
            required.insert(0, "title")
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    def _metadata_response_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "characters": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                        },
                        "required": ["name", "description", "role"],
                    },
                },
                "setting": {"type": "string"},
            },
            "required": ["summary", "characters", "setting"],
        }

    def _ending_review_response_schema(self, include_moral: bool = True) -> dict:
        """Structured semantic review for the final page only."""
        boolean_fields = {
            "resolves_opening_promise": {"type": "boolean"},
            "resolves_main_problem": {"type": "boolean"},
            "emotional_payoff_complete": {"type": "boolean"},
            "callback_earned": {"type": "boolean"},
            "no_new_plot": {"type": "boolean"},
            "ending_feels_earned": {"type": "boolean"},
            "satisfying_ending": {"type": "boolean"},
        }
        if include_moral:
            boolean_fields["moral_visible_through_action"] = {"type": "boolean"}
        return {
            "type": "object",
            "properties": {
                **boolean_fields,
                "reason": {"type": "string"},
                "required_changes": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            },
            "required": [*boolean_fields.keys(), "reason", "required_changes"],
        }

    def _canon_ending_review_response_schema(self) -> dict:
        """Structured semantic review for a Canon final page."""
        boolean_fields = {
            "canonical_ending_complete": {"type": "boolean"},
            "required_final_events_present": {"type": "boolean"},
            "required_event_order_preserved": {"type": "boolean"},
            "canonical_characters_preserved": {"type": "boolean"},
            "no_invented_resolution": {"type": "boolean"},
            "child_does_not_change_outcome": {"type": "boolean"},
            "no_unfinished_canon_event": {"type": "boolean"},
            "satisfying_canonical_close": {"type": "boolean"},
        }
        return {
            "type": "object",
            "properties": {
                **boolean_fields,
                "reason": {"type": "string"},
                "required_changes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 6,
                },
            },
            "required": [*boolean_fields.keys(), "reason", "required_changes"],
        }

    def _model_candidates(self) -> list[str]:
        """Return configured Gemini model plus safe fallbacks for transient deprecation/routing errors.

        This is only used when a model endpoint returns NOT_FOUND/404. The
        configured model remains first choice, so Render/local env settings
        still control normal production behaviour.
        """
        configured = (self.model_name or "").strip()
        candidates: list[str] = []
        for model in (
            configured,
            os.getenv("GEMINI_FALLBACK_MODEL", "").strip(),
            "gemini-3.5-flash",
            "gemini-2.5-flash-lite",
        ):
            if model and model not in candidates:
                candidates.append(model)
        return candidates

    def _generate_content_once_sync(
        self,
        model_name: str,
        prompt: str,
        response_schema: Optional[dict] = None,
        max_output_tokens: Optional[int] = None,
    ):
        return self.client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=self._json_generation_config(
                response_schema,
                max_output_tokens=max_output_tokens,
            ),
        )

    def _generate_content_sync(
        self,
        prompt: str,
        response_schema: Optional[dict] = None,
        max_output_tokens: Optional[int] = None,
    ):
        """Generate content using google.genai, preferring structured JSON output.

        Falls back once without response_schema if the schema/config is rejected.
        If the configured model endpoint returns NOT_FOUND/404, try safe model
        fallbacks instead of leaving the story permanently partial.
        """
        if not self.client or not self.model_name:
            raise ValueError("Gemini client is not configured")

        last_exc: Optional[Exception] = None
        for model_name in self._model_candidates():
            try:
                if model_name != self.model_name:
                    print(f"[PERF] genai_model_fallback_try model={model_name}")
                return self._generate_content_once_sync(
                    model_name,
                    prompt,
                    response_schema,
                    max_output_tokens,
                )
            except Exception as exc:
                last_exc = exc
                err = str(exc)
                is_not_found = "404" in err or "NOT_FOUND" in err or "not found" in err.lower() or "no longer available" in err.lower()

                # If the schema/config was rejected, keep the existing one-shot
                # retry without schema. Do not waste the no-schema retry on a
                # missing model; try the next model candidate instead.
                if response_schema and not is_not_found:
                    print(f"[PERF] genai_schema_call_failed retrying_without_schema error={err[:200]}")
                    return self._generate_content_once_sync(
                        model_name,
                        prompt,
                        None,
                        max_output_tokens,
                    )

                if is_not_found:
                    print(f"[PERF] genai_model_not_found model={model_name} trying_next error={err[:200]}")
                    continue

                raise

        raise last_exc or ValueError("Gemini generation failed for all model candidates")

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
            "dragons": {"en": "dragons", "es": "dragones", "fr": "dragons", "de": "Drachen", "it": "draghi", "ja": "ドラゴン", "ar": "التنانين"},
            "space": {"en": "space", "es": "espacio", "fr": "espace", "de": "Weltraum", "it": "spazio", "ja": "宇宙", "ar": "الفضاء"},
            "animals": {"en": "animals", "es": "animales", "fr": "animaux", "de": "Tiere", "it": "animali", "ja": "動物", "ar": "الحيوانات"},
            "princess": {"en": "princess", "es": "princesa", "fr": "princesse", "de": "Prinzessin", "it": "principessa", "ja": "お姫さま", "ar": "الأميرة"},
            "adventure": {"en": "adventure", "es": "aventura", "fr": "aventure", "de": "Abenteuer", "it": "avventura", "ja": "冒険", "ar": "المغامرة"},
            "underwater": {"en": "underwater", "es": "bajo el agua", "fr": "sous l’eau", "de": "Unterwasserwelt", "it": "mondo sottomarino", "ja": "海の中", "ar": "تحت الماء"},
            "forest": {"en": "forest", "es": "bosque", "fr": "forêt", "de": "Wald", "it": "foresta", "ja": "森", "ar": "الغابة"},
            "magic": {"en": "magic", "es": "magia", "fr": "magie", "de": "Magie", "it": "magia", "ja": "魔法", "ar": "السحر"},
            "dinosaurs": {"en": "dinosaurs", "es": "dinosaurios", "fr": "dinosaures", "de": "Dinosaurier", "it": "dinosauri", "ja": "恐竜", "ar": "الديناصورات"},
            "superheroes": {"en": "superheroes", "es": "superhéroes", "fr": "super-héros", "de": "Superhelden", "it": "supereroi", "ja": "スーパーヒーロー", "ar": "الأبطال الخارقون"},
            "emotions": {"en": "emotions", "es": "emociones", "fr": "émotions", "de": "Gefühle", "it": "emozioni", "ja": "気持ち", "ar": "المشاعر"},
        }
        return theme_labels.get(key, {}).get(lang) or theme_labels.get(key, {}).get("en") or raw

    def _localized_relationship_label(self, relationship: Optional[str], language_code: Optional[str]) -> str:
        raw = str(relationship or "").strip()
        if not raw:
            return raw

        key = raw.lower().replace("-", "_").replace(" ", "_")
        lang = (language_code or "en").lower()[:2]
        relationship_labels = {
            "mother": {"en": "mother", "es": "madre", "fr": "mère", "de": "Mutter", "it": "mamma", "ja": "お母さん", "ar": "الأم"},
            "mum": {"en": "mum", "es": "mamá", "fr": "maman", "de": "Mama", "it": "mamma", "ja": "お母さん", "ar": "ماما"},
            "mom": {"en": "mum", "es": "mamá", "fr": "maman", "de": "Mama", "it": "mamma", "ja": "お母さん", "ar": "ماما"},
            "father": {"en": "father", "es": "padre", "fr": "père", "de": "Vater", "it": "papà", "ja": "お父さん", "ar": "الأب"},
            "dad": {"en": "dad", "es": "papá", "fr": "papa", "de": "Papa", "it": "papà", "ja": "お父さん", "ar": "بابا"},
            "sister": {"en": "sister", "es": "hermana", "fr": "sœur", "de": "Schwester", "it": "sorella", "ja": "姉妹", "ar": "الأخت"},
            "brother": {"en": "brother", "es": "hermano", "fr": "frère", "de": "Bruder", "it": "fratello", "ja": "兄弟", "ar": "الأخ"},
            "friend": {"en": "friend", "es": "amigo o amiga", "fr": "ami ou amie", "de": "Freund oder Freundin", "it": "amico o amica", "ja": "友だち", "ar": "صديق أو صديقة"},
            "cat": {"en": "cat", "es": "gato", "fr": "chat", "de": "Katze", "it": "gatto", "ja": "猫", "ar": "قطة"},
            "dog": {"en": "dog", "es": "perro", "fr": "chien", "de": "Hund", "it": "cane", "ja": "犬", "ar": "كلب"},
            "pet": {"en": "pet", "es": "mascota", "fr": "animal de compagnie", "de": "Haustier", "it": "animale domestico", "ja": "ペット", "ar": "حيوان أليف"},
        }
        return relationship_labels.get(key, {}).get(lang) or relationship_labels.get(key, {}).get("en") or raw

    def _no_companion_required_text(self, language_code: Optional[str]) -> str:
        return {
            "en": "No companion is required.",
            "es": "No hace falta incluir ningún compañero.",
            "fr": "Aucun compagnon n’est nécessaire.",
            "de": "Es muss kein Begleiter vorkommen.",
            "it": "Non è necessario includere un compagno.",
            "ja": "物語に相棒を登場させる必要はありません。",
            "ar": "لا حاجة إلى إضافة رفيق في القصة.",
        }.get((language_code or "en").lower()[:2], "No companion is required.")

    def _no_extra_characters_required_text(self, language_code: Optional[str]) -> str:
        return {
            "en": "No extra family members or friends are required.",
            "es": "No hace falta incluir familiares, amistades ni mascotas adicionales.",
            "fr": "Aucun membre de la famille, ami ou animal supplémentaire n’est nécessaire.",
            "de": "Es müssen keine zusätzlichen Familienmitglieder, Freunde oder Haustiere vorkommen.",
            "it": "Non è necessario includere altri familiari, amici o animali.",
            "ja": "家族や友だち、ペットを追加で登場させる必要はありません。",
            "ar": "لا حاجة إلى إضافة أفراد آخرين من العائلة أو الأصدقاء أو الحيوانات الأليفة.",
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
- Reading/listening stage: Oxford Reading Tree Stage 5-7 inspired clarity, not a children's novel.
- The child should understand almost every sentence the first time they hear it.
- If story quality and reading level conflict, ALWAYS choose the lower reading level.
- Sentence shape: mostly 5-10 words. Use one clear idea per sentence.
- Vocabulary: everyday spoken words first. Prefer simple verbs: looked, saw, got, went, made, took, put, asked, tried, helped, shared, fixed.
- Use only a few richer story words in the whole story, not every page. Good examples: clue, promise, bridge, brave, patient, puzzled.
- Avoid poetic narration, literary adjectives, symbolic language, and abstract emotional phrases.
- Dialogue: short and plain. It should sound like real words a six-year-old can follow.
- Plot: one main goal, one main helper, one obstacle, and one simple first idea that may not work.
- Emotion: worried, shy, sad, brave, proud, patient. Show through small actions and simple dialogue.
- Humour: clear visual mishaps and simple misunderstandings that affect the plot.
- Parents should think: "This sounds simple and clear," not "This sounds beautifully written."
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
- Use simple early-reader adventure language that a tired six-year-old can follow.
- Most sentences should be 5-10 words. A longer sentence is allowed only if it is still very easy.
- One sentence should usually contain one action or one idea.
- Use familiar vocabulary first. Richer story words are allowed rarely, and only when meaning is obvious.
- Use one clear goal, one main helper, and one main obstacle.
- Use no more than 3 important characters.
- Funny moments should be visual and easy to understand.
- Do not introduce several new characters, places, objects, and problems on the same page.
- Avoid advanced, poetic, or abstract phrases such as "silent circus of clouds", "silver acrobats", "balanced on moonbeams", "belonged to the great Star Ringmaster", "the village had lost hope", or "the courage inside her heart".
- Avoid story language that sounds like age 8-10 fiction. This age needs clear, simple, concrete wording.
- If a sentence sounds beautiful but hard, rewrite it in plain words.
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
- Use vocabulary that feels like early independent reading and easy read-aloud comprehension.
- Prefer common concrete words: look, see, find, help, make, take, give, open, close, hold, carry, try, ask, say, go, come, stop, wait, share, fix, friend, door, path, bridge, river, tree, room, garden, castle, dragon.
- Allow only a few gentle story words in the whole story: clue, promise, puzzled, patient, brave, hidden.
- Avoid using richer words on every page. One simple new word is enough.
- Prefer clear concrete verbs over adult or abstract verbs: looked, saw, asked, tried, carried, opened, helped.
- Avoid older vocabulary such as investigate, responsibility, extraordinary, magnificent, peculiar, complicated, astonished, remarkable, consequence, tradition, official, cautious, suspicious, determined, solution, festival, invitation.
- Avoid poetic or abstract phrases like “the village had lost hope”, “the courage inside her heart”, “a symbol of belonging”, “the silence folded around her”, or “hope rose in the room”.
- Avoid figurative language unless a six-year-old can picture it immediately.
- Word budget: about 95% familiar words, 5% gentle new vocabulary.
- Sentence rhythm: short, plain sentences with one clear action.
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
- Each page must clearly move the story forward. It may arrive somewhere, reveal something, test an idea, force a choice, create a setback, solve part of the problem, or settle after the result. These are examples, not a fixed sequence.
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

    def _literary_polish_rules(self) -> str:
        return """PILLOWTALES LITERARY POLISH — STRUCTURAL, NOT DECORATIVE:
- Keep the existing clear plot. Do not add complexity merely to sound polished.
- Show important feelings through a visible action, body response, choice, or short line of dialogue.
- Avoid repeatedly naming emotions with lines such as "felt brave", "felt proud", "felt happy", or "was very worried".
- For younger children, use simple concrete behaviour to show feeling, but vary it naturally. Do not default repeatedly to tummy flutters/flip-flops, widened eyes, gasps, smiles, warm feelings in the chest, heart-sinking, wobbling knees, or similar stock reactions.
- Give the story one unforgettable child-friendly moment: a visual mishap, funny misunderstanding, odd habit, or magical detail that affects what happens next.
- Give one important side character a recognisable behaviour that appears at least twice and influences the plot.
- Vary sentence openings. Do not begin several consecutive sentences with the child's name or the same pronoun.
- Let dialogue breathe. A short spoken line may stand alone before the next action.
- Use sensory detail sparingly and concretely: one sound, texture, smell, taste, or movement when it helps the scene.
- Bring back one early detail later as a useful or emotional callback.
- The moral must remain inside the action. Do not explain it.
- The last page must not introduce a new named friend, helper, creature, location, object, or secret.
- End on a concrete image, action, sound, or callback from this story rather than a summary of how the child felt.
- A strong final story sentence lets the listener picture or hear something already established: a wave, chirp, ribbon, crown, lantern, blanket, laugh, pawprint, feather, star, or other story-specific detail.
- Do not end the story with generic achievement wording such as "felt like the bravest...", "was very proud", "was ready for anything", or "had learned an important lesson".
"""

    def _ending_engine_rules(self) -> str:
        return """PHASE 12 ENDING ENGINE — NON-NEGOTIABLE:
- You are not starting a new scene. You are finishing the exact story already written.
- Everything on the final page must follow directly from the existing goal, problem, characters, setting, promises, jokes, and objects.
- Do not change direction or replace the story with a generic bedtime ending.
- Resolve the main external problem using actions, choices, relationships, clues, or rules already established.
- Then show the emotional payoff through what the child or supporting character says or does.
- Reuse one meaningful earlier detail: a promise, object, phrase, joke, comfort habit, helper behaviour, relationship moment, or world rule.
- The callback must feel earned. It should solve, comfort, repair, reward, or quietly show change.
- Let the characters briefly enjoy the result. This afterglow should be part of this story, not a generic celebration.
- Bring the child to a clearly safe and settled place already established or naturally connected to the story.
- After the resolution, slow the pace with simple actions and fewer details.
- Use one fitting bedtime image only when it belongs naturally in this story.
- Do not explain the moral. Never write that the child learned, realised, understood, or remembered the lesson.
- Do not introduce a new character, location, task, conflict, clue, magical object, mystery, or promise.
- Never finish with another adventure waiting, a door opening, a clue appearing, a sound in the distance, a character promising to return, or any invitation to continue.
- The final story sentence must be specific to this story, easy to read aloud, emotionally complete, and impossible to mistake for a mid-story sentence.
- End on a concrete image, action, sound, or callback involving something already established in the story.
- Do not end on a summary of the child's emotion or achievement.
- Never use a final sentence such as "felt like the bravest...", "was very proud", "was ready for anything", or another generic self-description.
- After that final story sentence, write exactly: The End.

FINAL ENDING CHECKLIST — SILENTLY VERIFY ALL:
- the original main problem is resolved
- the resolution grows from earlier events
- emotional reward is shown
- one earlier detail returns naturally
- no new plot thread appears
- the child and important characters are safe and settled
- the pace slows after the resolution
- the last story sentence is a concrete image, action, sound, or callback from this specific story
- the last story sentence is not a generic feeling or achievement statement
- no moral explanation
- no sequel hook
- no unanswered question
- the page ends exactly with The End.
"""

    def _story_spine_block(
        self,
        request: GenerateStoryRequest,
        title: Optional[str] = None,
        first_page: Optional[str] = None,
    ) -> str:
        title_line = title or "the story title created with Page 1"
        opening_text = (first_page or "").strip()
        opening_excerpt = opening_text[:1200] if opening_text else (
            "Page 1 must establish one clear wish, problem, question, promise, or emotional need."
        )

        if self._is_folk_adventure_request(request):
            seed = self._select_living_world_episode_seed(request) or {}
            return f"""FIXED LIVING WORLD EPISODE SPINE — DO NOT DRIFT:
- Story title: {title_line}
- Selected episode seed: {json.dumps(seed, ensure_ascii=False)}
- Page 1 establishes the permanent episode promise:
  {opening_excerpt}
- Keep the same protagonist and central world-specific problem from page to page.
- Every major event must deepen, complicate or resolve that problem.
- The listening child never appears in the episode.
- Do not replace the premise with an object-retrieval, memory, wish, shell, ribbon, portal or moral demonstration plot.
- By Page 4, the story must have materially changed direction or understanding through a meaningful clue, reveal, decision or reversal.
- Page 5 must contain the strongest genuine setback, failed attempt, difficult choice or complication. Do not let the protagonist walk directly from discovery to easy success.
- Page 6 delivers the decisive action and climax. The protagonist must drive it.
- Page 7 may complete the already-established climax when needed, but it must also provide consequence, emotional payoff, an earned callback and a safe settled close.
- Never save the basic cause, culprit or entire solution for an unexplained last-page invention.
"""

        moral_lines = (
            f"""- Requested moral: {request.moral}
- The requested moral must remain recognisable through the child's choices and consequences, but must never be lectured or repeatedly named.
- At least one meaningful choice in the middle and the decisive action near the ending must demonstrate the moral."""
            if self._moral_requested(request)
            else "- No moral was requested. Do not invent, demonstrate or validate one."
        )

        return f"""FIXED STORY SPINE — DO NOT DRIFT:
- Story title: {title_line}
- Theme: {request.customTheme or request.theme}
{moral_lines}
- Page 1 is the permanent opening promise of this story:
  {opening_excerpt}
- Identify the ONE central wish, problem, question, promise, or emotional need established there and keep it active from page to page.
- Every major event must deepen that same problem, reveal something important about it, test the child's choices, or move directly toward its resolution.
- Do not replace the opening promise with a more convenient object-retrieval, repair, delivery, celebration, or generic magical-task plot unless Page 1 clearly established that plot.
- Preserve one simple Page 1 detail as an ending callback: an object, phrase, wish, sound, promise, joke, helper behaviour, or image.
- The child must make at least one meaningful choice that changes what happens next.
- Page 6 must bring the original problem to its decisive resolution or make that resolution inevitable.
- Page 7 must answer or fulfil the opening promise, complete the emotional change, reuse the callback naturally, and settle safely.
- The ending is incomplete if it only fixes an object or completes a task while leaving the opening wish, question, relationship, or emotional need unanswered.
- The ending should depend on specific events, choices, relationships, clues, habits, or rules from this story. If it could be pasted into another story with only the names changed, it is not good enough.
"""

    def _first_page_spine_setup_rules(self, request: GenerateStoryRequest) -> str:
        moral_line = (
            f"- Connect it naturally to the requested moral: {request.moral}."
            if self._moral_requested(request) else
            "- No moral is requested. Do not invent a moral requirement."
        )
        return f"""STORY SPINE SETUP:
- Establish exactly one clear opening promise, problem, opportunity or conflict.
- Make it strong enough to guide all seven pages.
{moral_line}
- Plant one simple detail that can return meaningfully in the ending.
- Do not solve it on Page 1.
- Do not create several unrelated mysteries, objects, destinations or goals.
"""

    def _story_flow_rules(self) -> str:
        """Surgical anti-repetition guidance with persistent moral visibility."""
        return """STORY FLOW AND MORAL RESTRAINT:
- Keep the opening promise and requested moral active from Page 1 to Page 7.
- The requested moral must remain recognisable through the protagonist's choices, consequences, relationships, and final resolution.
- Do not repeatedly explain, name, or restate the moral after each event.
- At least one meaningful choice in the middle and the decisive action near the ending must clearly demonstrate the moral.
- Characters should not stop to explain what the reader has just seen.
- Do not state the same discovery, interpretation, feeling, or moral twice in slightly different wording. Once the reader can infer it from action or dialogue, move the story forward.
- Show the moral through choices, dialogue, consequences, changed behaviour, and the final resolution.
- Conspicuous setup must pay off: if a detail is presented as unusual, important, magical, mysterious, promised, or memorable, make it useful later or do not emphasise it.
- Trust children to understand an obvious moral from what happens, but do not allow the moral to disappear from the plot.
- Do not make every encounter follow the same sequence of event, moral reminder, reward, and reset.
- If the same scene pattern has appeared twice, the next page MUST use a different kind of event: dialogue, discovery, setback, surprise, cooperation, a difficult choice, or a quiet emotional turn.
- Never write three consecutive pages built around the same encounter shape, such as meeting a new character, hearing a similar problem, and moving on.
- The central conflict must appear no later than Page 3.
- Pages 4 and 5 must deepen that same conflict rather than replacing it with a different problem.
- Repeated phrases, counts, glowing objects, rewards, or symbolic reactions may return only when they move the plot or create a meaningful callback.
- A child should remember the adventure first and understand the moral through it, without the narrator explaining it word by word.
"""

    def _page_narrative_role(self, page_number: int) -> str:
        """Return the single structural job for the next continuation page.

        This is prompt-only and keeps the existing Page-1-first architecture,
        one-page background batches, storage flow, narration, and timing intact.
        """
        roles = {
            2: """PAGE 2 ROLE — DEEPEN THE PROMISE:
- Deepen the exact wish, problem, question, promise, or emotional need from Page 1.
- Introduce only one useful helper, clue, obstacle, or world rule.
- End with a clear complication, decision, or next step.
- Do not begin a repeated tour of characters or places.
""",
            3: """PAGE 3 ROLE — FIRST REAL COMPLICATION:
- Let the first attempt partly fail, create a new difficulty, or reveal that the problem is not as simple as it looked.
- The complication must grow from Pages 1-2, not from a new subplot.
- Do not repeat Page 2's scene shape.
- The child must notice, ask, choose, or try something that affects what happens next.
""",
            4: """PAGE 4 ROLE — MIDPOINT TURN:
- Change the direction or understanding of the story.
- Reveal one unexpected, funny, magical, or emotional truth that makes the child rethink the plan.
- The child must make a meaningful decision here that changes what happens next.
- If a moral was requested, begin earning it through this decision or its consequence; do not merely mention the moral.
- Bring forward an established clue, relationship, rule, promise, habit, or themed element rather than inventing a convenient solution.
- Do not add another similar helper encounter, collection stop, or repeated version of the same problem.
""",
            5: """PAGE 5 ROLE — STRONGEST SETBACK:
- Create the greatest obstacle, setback, or difficult choice in the story.
- Success should briefly feel uncertain, but never frightening.
- Do not solve the main problem on this page.
- The child should pause, test an idea, or wonder what to do next before becoming ready to act.
- Bring back one earlier clue, promise, habit, object, joke, relationship, themed element, or world rule and make it matter.
- If a moral was requested, the protagonist's response to this setback must set up the moral-in-action choice that will drive Page 6. Do not leave the moral for the ending to explain or repair.
- Everything introduced here must already belong to the existing story.
- Do not introduce another quest, helper, mystery, location, or magical rule.
- End with the child ready to take the decisive action on Page 6.
""",
            6: """PAGE 6 ROLE — DECISIVE ACTION AND CLIMAX:
- The child must take the decisive action. Helpers may contribute, but they must not solve it for the child.
- If a moral was requested, the decisive action itself must demonstrate it through a real choice and consequence. Do not rely on another character performing the moral at the end.
- If no moral was requested, resolve the problem through the protagonist's established choices, skills, relationships, clues, or story rules without inventing a lesson.
- Resolve the main external problem here, or make the final consequence inevitable using only established material.
- Include the story's most memorable moment: magical, funny, surprising, or emotional.
- This page must feel like the peak of the adventure, not merely preparation or another attempt.
- Leave Page 7 only for the completed result, emotional payoff, callback, and calm settling.
""",
            7: """PAGE 7 ROLE — PAYOFF AND BEDTIME LANDING:
- Begin after the decisive action, with the main problem already resolved or visibly completing.
- Show the emotional and relationship payoff through action or dialogue.
- Reuse one earned callback from Pages 1-6.
- Let the characters briefly enjoy the result, then slow the pace and settle safely.
- Do not add a new surprise, task, character, object, place, or problem.
""",
        }
        return roles.get(page_number, "")

    def _page_tension_rules(self, page_number: int) -> str:
        """Control narrative escalation without changing the selected plot."""
        rules = {
            2: """PAGE 2 TENSION:
- The situation should feel slightly more difficult or more important than Page 1.
- Introduce or sharpen the central conflict if Page 1 only hinted at it.
- Do not solve anything yet.
""",
            3: """PAGE 3 TENSION:
- The child's first approach should only partly work, fail safely, or reveal a larger difficulty.
- The central conflict must now be unmistakably clear.
- Raise the stakes through consequences already connected to the opening promise.
""",
            4: """PAGE 4 TENSION:
- Change the direction or understanding of the story.
- Reveal something unexpected that forces the child to reconsider the plan.
- The child must make an important decision that affects the remaining pages.
""",
            5: """PAGE 5 TENSION:
- This is the emotional low point and strongest setback.
- Success should briefly feel uncertain, but never frightening.
- Do not solve the main problem here.
- Do not introduce a new idea merely to create tension.
""",
            6: """PAGE 6 TENSION:
- Bring every important story thread toward the decisive action.
- The child should succeed because of choices, relationships, clues, habits, or learning established earlier.
- This is the peak; do not postpone the solution to Page 7.
""",
            7: """PAGE 7 TENSION:
- Add no new tension.
- Let the story release its energy and settle.
- Focus on payoff, callback, safety, and calm completion.
""",
        }
        return rules.get(page_number, "")

    def _natural_name_pronoun_rules(self, protect_canon_names: bool = False) -> str:
        """Global natural name/pronoun guidance for all story-generation paths.

        Prompt-only: this changes prose guidance, not stored names, Canon facts,
        narration, pronunciation, page flow, polling, or post-processing.
        """
        canon_rules = ""
        if protect_canon_names:
            canon_rules = """
CANON NAME PROTECTION:
- Pronouns may replace unnecessary repetition, but they never change Canon identity.
- Whenever a protected Canon name is actually written, preserve its stored spelling, accents, spacing and capitalisation exactly.
- Do not invent nicknames, anglicise, de-accent, respell, modernise or arbitrarily shorten protected names.
- A shortened Canon form is allowed only when the existing Canon rules explicitly permit that exact form.
"""

        return """NATURAL NAME & PRONOUN USAGE — GLOBAL STORYTELLING RULE:
- Introduce the protagonist clearly by name.
- Once the protagonist's identity is established, avoid unnecessarily repeating the name in consecutive or closely spaced sentences.
- When it is obvious who is acting, prefer the natural pronoun for that character rather than repeating the name.
- Reintroduce the protagonist's name when needed for clarity, after another character becomes the focus, following a scene or viewpoint transition, when pronouns could be ambiguous, or when the name adds genuine emotional or narrative emphasis.
- Keep personalisation present where the story mode allows it, but make it sound like natural human storytelling rather than repeated name insertion.
- Do not use a numeric name-frequency limit. Choose names or pronouns according to clarity, rhythm and natural prose.
- Do not mechanically replace names in post-processing; write the sentence naturally in the first place.
- Avoid several consecutive sentences beginning with the same name or the same pronoun when natural sentence variation is possible.
""" + canon_rules

    def _children_author_voice_rules(self) -> str:
        """Positive author-quality guidance for standard personalised stories.

        Prompt-only. This does not imitate any named writer and does not touch
        Story Worlds, narration, chunking, polling, reader state or storage.
        """
        return """CHILDREN'S AUTHOR VOICE — 9.8 QUALITY TARGET:
- Write with the warmth, confidence, wit, rhythm and specificity of an experienced children's author, while remaining completely original. Do not imitate or mention any named author.
- The prose must feel deliberately written for this particular story, not assembled from familiar children's-story phrases.
- Give the protagonist and important side characters personality through what they say, notice, misunderstand, avoid, attempt and choose. Let them occasionally hesitate, guess wrongly, change their mind or make a small harmless mistake.
- Prefer concrete, story-specific observations over generic fantasy decoration. One unusual detail that belongs to this adventure is better than several interchangeable glowing, sparkling or magical descriptions.
- Vary sentence rhythm naturally. Mix short punchy lines, dialogue, and longer clear sentences appropriate to the child's reading age. Do not make every paragraph follow the same sentence pattern.
- Dialogue should sound spoken. Characters may interrupt, disagree gently, joke, misread a situation, or answer imperfectly. Avoid dialogue whose only purpose is to explain the plot to the reader.
- Trust the reader. If an action, pause, expression, consequence or line of dialogue already communicates the feeling or lesson, do not explain it again.
- Let quiet moments count. Not every page needs a magical reveal, glowing object, new helper or spectacle. A choice, conversation, failed idea, observation or small act can carry a scene.
- Hide the story framework. The reader must never feel an obvious template of setup, clue, helper, setback, climax and moral, even though the story remains structurally complete.
- Include at least one memorable line, comic beat, behaviour, image or callback that could only belong to this story and that a child might repeat the next day.
- Avoid formulaic generated-story commentary such as 'This was definitely...', 'This was going to be...', 'With a determined breath...', 'her eyes twinkled', 'a little flutter of excitement', 'she realised that...', or close equivalents unless the wording is genuinely necessary and fresh.
- Never announce the child's numerical age in the story prose. Age is an internal writing calibration only.
"""

    def _standard_bedtime_first_page_quality_rules(self, request: GenerateStoryRequest) -> str:
        """Compact quality guard for the speed-critical standard Page 1 call."""
        moral_line = (
            f"- Moral: {request.moral}. Set up a situation where the protagonist can demonstrate it later through a choice; do not explain it now."
            if self._moral_requested(request)
            else "- No moral was requested. Do not invent one."
        )
        return f"""PAGE 1 QUALITY GUARD:
- The child's age is internal calibration only. Never state or hint at the numerical age in the story prose.
- Treat the selected theme as a promise: introduce its central subject or a direct path to it now.
- Establish one clear want, problem, question, opportunity or tension that can carry the story.
- Plant one useful, story-specific detail that can pay off later; avoid conspicuous details with no purpose.
- Keep the protagonist active. They should notice, decide, attempt, question or choose rather than simply being led from clue to clue.
- Give Page 1 a natural human hook, not a generic magical-object reveal by default.
{moral_line}
- Use natural names/pronouns, plain prose only, and no Markdown or formatting symbols.
- Avoid stock AI reactions, explanatory emotion, decorative simile overload and formulaic phrases such as 'This was going to be...' or 'With a determined breath...'.
"""

    def _standard_bedtime_elite_quality_rules(self, request: GenerateStoryRequest) -> str:
        """Focused quality rules for standard personalised Bedtime Stories only.

        Prompt-only. This deliberately does not touch Story Worlds, narration,
        chunking, polling, page ownership, reader state, storage or subscriptions.
        """
        moral_rule = (
            f"- Requested moral: {request.moral}. Build it into a meaningful choice by the protagonist before the final page; the ending must show the consequence, not explain the lesson."
            if self._moral_requested(request)
            else "- No moral was requested. Do not invent or force a lesson."
        )
        return f"""STANDARD BEDTIME STORY — ELITE QUALITY GUARD:
- The child's numerical age must never appear in the story prose. Use it only to calibrate vocabulary, sentence rhythm, inference, emotional depth, humour and plot complexity.
- The story must feel intentionally authored rather than generated. Use specific observations, natural rhythm, characterful dialogue and small human imperfections instead of polished generic storybook phrasing.
- Do not make the world hand the protagonist every next clue. At least one important connection should be worked out, tested or inferred by the protagonist.
- Allow a harmless wrong guess, hesitation, failed attempt or change of mind when it makes the protagonist feel more real and strengthens the eventual solution.
- Not every page needs spectacle or a new magical element. Quiet conversation, observation, choice, embarrassment, humour or practical problem-solving can carry a scene.
- Keep the seven-page architecture invisible. Do not use repeated transition language or narrator commentary that makes the structural role of each page obvious.
- Deliver the child's selected theme as a real story promise, not decorative scenery. The central subject of the theme must materially affect the plot and interact meaningfully with the protagonist early enough to matter.
- By the end of Page 2, the reader should understand what the protagonist wants, needs to solve, discover, help with, or decide. Keep that central question active until it is resolved.
- Do not hand the solution to the protagonist through a convenient adult, magical object, sudden gift, or unexplained coincidence. Helpers may offer information or opportunities, but the protagonist must notice, choose, try, connect, share, create, persist, or otherwise materially influence the outcome.
- Important setup must earn a payoff. A conspicuous smell, phrase, object, rule, mystery, promise, joke, clue, habit, or magical detail should either matter later or be removed. Do not create accidental mysteries that the story forgets.
- Trust the reader. When dialogue, action, expression, or consequence already shows an emotion or meaning, do not immediately explain the same thing again. Do not state the same discovery twice in slightly different words.
- Vary emotional reactions. Do not repeatedly use tummy flutters/flip-flops, widened or sparkling eyes, warm feelings in the chest, gasps, smiles, shoulder slumps, heart-sinking, or similar stock reactions as a default emotional vocabulary. Use dialogue, choices, pauses, movement, humour, and concrete behaviour instead.
- Use imagery selectively. One memorable comparison is stronger than several decorative similes on every page. Avoid an AI-storybook rhythm of constant twinkling, dancing, glowing, sparkling, tiny, warm, magical descriptions.
- Dialogue should sound like characters speaking to one another, not the narrator delivering instructions. Keep dialogue and its attribution together naturally; never split a speech from a short attribution in a way that reads like broken prose.
- The protagonist must make at least one meaningful mid-story decision and must drive the decisive action near the ending. The resolution must not simply happen around them.
{moral_rule}
- Do not postpone a story-wide moral or character requirement until Page 7. Pages 4-6 must already contain the choice/action that earns the ending.
- The final page completes the promise of the title/theme and the central story question. Do not introduce the most important themed character, relationship, destination, or activity only at the end.
- The ending should let the characters briefly experience the result, then settle. Never end with 'the visit was just beginning', 'another adventure was waiting', or another opening disguised as a conclusion.

PLAIN PROSE — NON-NEGOTIABLE:
- Story text must be plain prose only. Never use Markdown or formatting notation inside story content.
- Never use *emphasis*, **bold**, _formatting_, __formatting__, headings, bullet markers, code backticks, fenced code blocks, or other presentation markup.
- Express emphasis through natural wording, punctuation, dialogue, or sentence rhythm instead.
"""

    def _storycraft_rules(self) -> str:
        return """PILLOWTALES STORY VOICE:
- Write as a skilled children's storyteller with a confident, warm, playful and natural read-aloud voice.
- Write stories children ask to hear again tomorrow night and parents enjoy reading aloud.
- The goal is not decorative cleverness. The goal is curiosity, character, humour, feeling and one story idea the child remembers tomorrow.
- Make the prose feel intentionally authored and story-specific, with natural rhythm and personality rather than generic storybook polish.

CORE STORY PROMISE:
- Tell a simple, warm, funny bedtime adventure that a child can understand and remember tomorrow.
- Give the child something worth listening to: a clear goal, a real problem, a useful surprise, and a satisfying result.
- Every story needs one clear main idea that can be said in one sentence.
- Prefer memorable characters over complicated plots.
- Prefer a funny wish, mistake, job, fear, habit, or misunderstanding over epic danger.
- Good story energy: a taxi-driver lion who wants to drive a bus; a pigeon who wants to be an eagle; a sparrow who thinks it can lead the winter flight; a dragon who sneezes bubbles; a pirate whose map always finds cake.
- Harmless silliness is welcome when it grows naturally from character or situation and helps the child smile.
- Do not try to impress adults. Do not write cinematic, epic, literary, poetic, grand, or fantasy-novel prose.

SIGNATURE PILLOWTALES MAGIC:
- Include at least one memorable magical detail that belongs only to this story.
- The magical detail should be simple enough for a child to repeat the next day.
- Examples of the kind of detail wanted, not to copy: clouds that smell of pancakes, a dragon scared of butterflies, a moon that forgot bedtime, a squirrel post office, a bus full of penguins.
- The magical detail must affect the story, create a clue, solve a problem, cause a laugh, or return at the end.
- Use one clear magical idea per scene. Do not stack lots of magical objects, rules, titles, maps, and ancient secrets.

AGE-FIRST LANGUAGE RULES:
- Age-appropriate language is the highest priority. If a sentence sounds beautiful but too grown-up, simplify it.
- For ages 6 and under, use plain everyday words, short sentences, and concrete actions.
- For ages 6 and under, most sentences should be 5-10 words.
- For ages 6 and under, avoid abstract emotions, symbolic lessons, complex lore, and grown-up vocabulary.
- For ages 7-8, allow a little more detail, but keep the story clear and read-aloud friendly.
- For ages 9+, richer language is allowed, but never adult literary prose.
- Do not use older-child words just to sound polished.
- Prefer: said, asked, looked, saw, went, got, made, took, put, tried, helped, shared, fixed.
- Avoid unless truly needed: magnificent, extraordinary, mysterious, ancient, remarkable, consequence, responsibility, cautious, suspicious, determined, investigate.

STORY SHAPE RULES:
- Begin with who the story is about, where they are, and what makes today different.
- Pages 1-2 should create curiosity and establish a goal children can get behind.
- Pages 3-5 should contain real progress, one setback or surprise, and at least one choice made by the child.
- Page 6 should resolve the main challenge through the moral in action.
- Page 7 should fully complete the adventure, resolve the emotional need, and let the energy settle into a safe bedtime landing.
- Do not use an unresolved cliffhanger, sequel hook, invitation to continue, or unanswered question. The current story must feel complete.
- Keep one clear problem, wish, dream, misunderstanding, or task visible from page to page.
- Every page should move the story forward.
- End Pages 1-5 with a small reason to keep listening: a new clue, choice, surprise, funny complication, or next step.
- These hooks must create curiosity, not fear, panic, or an unresolved bedtime cliffhanger.
- Avoid pages that only search, wait, look around, or explain. Something should change on each page.
- Avoid crowded pages with several new characters, places, objects, and rules at once.
- Use short dialogue and action instead of narrator explanation.
- The child should help solve the problem through noticing, asking, choosing, trying, sharing, waiting, or being brave.
- Helpers may be funny, confused, or useful, but they must not solve everything.

HUMOUR AND WONDER RULES:
- Include gentle child-friendly humour when natural: funny misunderstandings, odd jobs, silly wishes, wrong hats, talking objects, or surprising but harmless behaviour.
- Humour should feel warm, not sarcastic, noisy, mean, embarrassing, or toilet-based.
- At least one funny or surprising moment should change what happens next.
- A parent should smile at least once while reading the story aloud.

MORAL RULES:
- Let kindness, bravery, sharing, patience, confidence, or friendship emerge naturally through what characters do.
- Demonstrate the requested moral through actions, choices, dialogue, and natural consequences rather than repeated explanation.
- Trust children to understand an obvious moral from the story. Do not explain it word by word.
- Once the moral direction is clear, do not restate it after each scene or encounter.
- Never explain the moral as a lesson.
- Avoid lines such as "Emily learned that...", "the lesson was...", or "everyone understood that...".
- Let the child feel the moral through the ending.

LANGUAGE CLEANUP RULES:
- Strongly avoid repeated AI-style words: gentle, tiny, little, small, golden, shimmering, glowing, sparkling, moonlit, softly, slowly, sleepy, magical, ancient, mysterious, magnificent, extraordinary.
- Use these words only when they are genuinely needed and age-appropriate.
- Avoid adverb-heavy writing, especially excitedly, carefully, suddenly, quickly, happily, softly.
- Prefer simple nouns, verbs, and dialogue.
- Do not describe the main child as little, small, tiny, young, brave little child, little girl, little boy, small hands, little feet, or young explorer.
- Refer to the child by name or pronouns. Treat the child as the capable hero.
- Do not invent physical descriptions of the child such as size, hair, eyes, clothes, or skin unless provided.

ENDING RULES:
- End warmly, specifically, and completely.
- Fully solve the main problem and close every important story question.
- Give the child an emotional reward shown through action, dialogue, or a changed relationship. Do not explain the feeling.
- Bring back one earlier funny detail, phrase, promise, object, helper habit, or world rule as an earned callback.
- Return the child to safety, home, family, bed, camp, village, or another clearly settled place appropriate to the story.
- Slow the rhythm on the final page. Use shorter, calmer actions after the resolution.
- Use one concrete bedtime-ready image such as a blanket, warm light, quiet room, moon, stars, soft rain, closed curtains, or a familiar object resting safely. Use only what fits naturally.
- End with one memorable final sentence that feels calm and complete.
- Do not add a sequel hook, invitation to return, new mystery, unexplained sound, opening door, blinking clue, or promise of another adventure.
- Do not explain the moral or use lines such as learned that, realised that, or understood that.
- Do not end with a generic line like everyone smiled happily, ready for sleep, or another adventure waited.
- The final page must end exactly with The End.

FINAL QUALITY CHECK:
- Before returning the story, silently ask: would a child understand it, want to hear the next page, smile at least once, and remember the main idea tomorrow?
- Check that every page moves the story forward and no page feels like filler.
- Check that the title is a real story title, never a placeholder such as Short title.
- If the story feels flat, add one useful funny moment or one memorable magical detail.
- If the language feels grown-up, simplify it."""

    def _select_opening_seed(self, request: GenerateStoryRequest) -> dict:
        """Select an age-safe and theme-aligned opening seed.

        This remains local and instant. It prevents mismatches such as a space
        story opening in a dinosaur kindergarten while preserving Page-1-first
        speed and architecture.
        """
        child = request.childName or "the child"
        language_code = (request.storyLanguageCode or "en").lower()[:2]
        theme_key = str(request.theme or request.customTheme or "").lower().replace("-", "_").replace(" ", "_")

        theme_seed_allowlist = {
            "space": {"ancient_observatory", "star_painter_cottage", "river_of_stars", "moon_bakery", "pillow_harbour"},
            "dragons": {"dragon_market", "dragon_post_office", "sleepy_castle_hall"},
            "dragon": {"dragon_market", "dragon_post_office", "sleepy_castle_hall"},
            "princess": {"glass_slipper_cafe", "sleepy_castle_hall", "hidden_garden_gate"},
            "animals": {"forest_school", "honeybee_palace", "little_lighthouse_cafe", "sleepy_forest_path"},
            "animal": {"forest_school", "honeybee_palace", "little_lighthouse_cafe", "sleepy_forest_path"},
            "forest": {"sleepy_forest_path", "forest_school", "hidden_garden_gate", "rainbow_garden"},
            "dinosaurs": {"dinosaur_kindergarten", "hidden_dinosaur_valley"},
            "dinosaur": {"dinosaur_kindergarten", "hidden_dinosaur_valley"},
            "underwater": {"underwater_palace", "mermaid_library", "whale_island", "seaside_cave"},
            "adventure": {"pillow_harbour", "glowing_attic", "hidden_garden_gate", "toymaker_workshop", "little_lighthouse_cafe"},
            "magic": {"moonlit_library", "toymaker_workshop", "rainbow_garden", "hidden_garden_gate"},
        }

        age_pool = self._seed_pool_for_age(request.age)
        allowed_for_theme = theme_seed_allowlist.get(theme_key)
        if allowed_for_theme:
            themed_pool = [seed for seed in age_pool if seed.get("family") in allowed_for_theme]
            if themed_pool:
                age_pool = themed_pool

        seed = random.choice(age_pool)
        template = seed.get(language_code)
        if not template and language_code == "ja":
            template = "{childName}は、眠る前の静かな時間に、小さな不思議を見つけました。"
        elif not template and language_code == "ar":
            template = "في هدوء ما قبل النوم، وجد {childName} شيئًا صغيرًا وعجيبًا ينتظر بداية الحكاية."
        elif not template:
            template = seed["en"]
        return {
            "family": seed.get("family", "place_entry"),
            "sentence": template.replace("{childName}", child),
        }

    def _opening_transition_rule(self, opening_family: str) -> str:
        return (
            f"The first sentence uses the '{opening_family}' place-entry opening. "
            "After that exact sentence, keep the story grounded in that place and move into the first gentle action. "
            "Do not drift into another abstract moon/window/glowing-light setup. "
            "Make the place feel like the doorway into the story."
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
- Use warm British bedtime storytelling with clear emotion, simple wonder, and child-friendly magic.
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
        if language_code == "ja":
            return "Write natural Japanese bedtime prose for a young child. Use clear native Japanese, short age-appropriate sentences, natural dialogue, and a warm read-aloud rhythm. Avoid literal translated phrasing."
        if language_code == "ar":
            return "Write natural Arabic bedtime prose for a young child. Use clear child-friendly Arabic, short age-appropriate sentences, natural dialogue, and a warm read-aloud rhythm. Avoid stiff or literal translated phrasing."
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
                "Include these family members, friends, or pets naturally if they fit the story. "
                "Do not overload the story with extra people. "
                "For pets, use ONLY the pet name and animal type provided; do not invent colour, breed, markings, size, collar, eye colour, or other physical details.\n"
                f"{characters_block}"
            )
        else:
            character_instruction = self._no_extra_characters_required_text(language_code)

        language_style_block = self._language_style_block(request.storyLanguageCode)
        story_world_block = self._story_world_prompt_block(request)
        standard_author_voice_block = (
            self._children_author_voice_rules()
            if not self._is_canon_request(request) and not self._is_folk_adventure_request(request)
            else ""
        )

        return f"""You are writing a PillowTales bedtime story.

LANGUAGE:
- Write ONLY in {language_name}.
- Do not mix languages.
- Write naturally for native-speaking children in {language_name}.
{language_style_block}

STORY FACTS:
- Child name: {request.childName}
- Internal reading age: {request.age}. Use this only for writing calibration; never state the numerical age in the story prose.
- Theme: {effective_theme}
- Moral: {request.moral}

STORY WORLD ISOLATION:
- If no Story World context follows, this is a standard PillowTales story.
- Do not import characters, places, canon, continuity, terminology or institutions from Story Worlds into a standard story.

{story_world_block}

{standard_author_voice_block}
{self._first_page_spine_setup_rules(request)}
{self._natural_name_pronoun_rules(protect_canon_names=self._is_canon_request(request) or self._is_folk_adventure_request(request))}
{self._storycraft_rules()}
{self._story_flow_rules()}
{self._literary_polish_rules()}
{self._ending_engine_rules()}
{self._age_readability_block(request.age)}
{self._age_vocabulary_block(request.age)}
{self._age_quality_control_block(request.age)}

PRODUCTION STORY CONTRACT:
- Create exactly 7 pages.
- Each page should have exactly 2 short paragraphs.
- Each page should contain about 5-7 read-aloud sentences.
- Each page should normally be 115-155 words.
- Use a clear beginning, middle, and ending.
- Make the story engaging enough that the child wants to keep listening, while keeping danger mild and bedtime-safe.
- Keep one main story idea visible from start to finish.
- Do not create one strong Page 1 followed by thin summary pages.
- The moral must be shown through what the child does, not explained.

COMPANION:
- {companion_line}

CHARACTERS:
- {character_instruction}

OUTPUT FORMAT STRICT:
Return ONLY valid JSON:
{{"title":"...","pages":["page 1 text","page 2 text","page 3 text","page 4 text","page 5 text","page 6 text","page 7 text"]}}

OUTPUT RULES:
- The pages array must contain exactly 7 strings.
- No markdown, notes, explanations, or extra keys.
- The final page must end exactly with The End."""

    def _intended_page_count(self, request: GenerateStoryRequest) -> int:
        return 7

    def _strip_json_fences(self, response_text: str) -> str:
        """Remove common markdown/code-fence wrapping without changing content."""
        cleaned = (response_text or "").strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:].strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
        return cleaned

    def _extract_json_object_text(self, response_text: str) -> str:
        """Return the most likely JSON object substring.

        Gemini sometimes returns whitespace, markdown fences, or a short note
        around JSON. This keeps normal valid JSON fast while tolerating harmless
        wrapper text. It does not invent missing story content.
        """
        cleaned = self._strip_json_fences(response_text)
        if not cleaned:
            raise ValueError("Empty Gemini response")

        if cleaned.startswith("{"):
            return cleaned

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No complete JSON object found in Gemini response")
        return cleaned[start:end + 1].strip()

    def _clean_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Gemini JSON with light wrapper cleanup only.

        This function intentionally avoids guessing missing closing strings or
        fabricating pages. Truncated responses are handled separately by retry
        and safe page salvage in the background continuation path.
        """
        json_text = self._extract_json_object_text(response_text)
        try:
            parsed = json.loads(json_text)
        except json.JSONDecodeError as exc:
            print(
                f"[PERF] json_parse_failed chars={len(response_text or '')} "
                f"error={str(exc)[:200]}"
            )
            raise

        if not isinstance(parsed, dict):
            raise ValueError("Gemini JSON root was not an object")
        return parsed

    def _extract_complete_strings_from_array_text(self, array_text: str) -> list[str]:
        """Extract only fully closed JSON strings from an array body.

        This is used for safe salvage when Gemini truncates after returning one
        or more complete page strings. A half-written final page is ignored.
        """
        decoder = json.JSONDecoder()
        values: list[str] = []
        idx = 0
        length = len(array_text or "")

        while idx < length:
            while idx < length and array_text[idx] in " \r\n\t,":
                idx += 1
            if idx >= length or array_text[idx] == "]":
                break
            if array_text[idx] != '"':
                break

            try:
                value, next_idx = decoder.raw_decode(array_text, idx)
            except json.JSONDecodeError:
                break

            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    values.append(stripped)
            idx = next_idx

        return values

    def _salvage_pages_from_response_text(self, response_text: str, max_pages: int) -> list[str]:
        """Safely recover complete page strings from malformed Gemini JSON.

        Safe means:
        - only reads strings that were fully closed JSON strings;
        - does not use incomplete trailing text;
        - postprocesses the same way as normal pages;
        - returns at most the requested batch size.
        """
        cleaned = self._strip_json_fences(response_text)
        pages_key = '"pages"'
        key_idx = cleaned.find(pages_key)
        if key_idx == -1:
            key_idx = cleaned.find("'pages'")
        if key_idx == -1:
            return []

        array_start = cleaned.find("[", key_idx)
        if array_start == -1:
            return []

        array_end = cleaned.find("]", array_start)
        if array_end == -1:
            array_body = cleaned[array_start + 1:]
        else:
            array_body = cleaned[array_start + 1:array_end]

        salvaged = self._extract_complete_strings_from_array_text(array_body)
        if not salvaged:
            return []

        pages = self._sanitize_generated_pages(postprocess_story_pages(salvaged))[:max_pages]
        print(
            f"[PERF] salvaged_pages_from_malformed_json count={len(pages)} "
            f"requested={max_pages} response_chars={len(response_text or '')}"
        )
        return pages

    def _count_story_sentences(self, text: str) -> int:
        """Best-effort multilingual sentence count for story-quality validation.

        Recognises Latin, Arabic and Japanese sentence-ending punctuation.
        This remains deliberately lightweight and local. It does not change
        Page-1-first architecture, narration, polling, subscriptions, Parent
        Voice, or reader behaviour.
        """
        cleaned = str(text or "").strip()
        if not cleaned:
            return 0
        parts = re.split(r'[.!?؟。！？]+(?:\s+|$)?', cleaned)
        return len([part for part in parts if part.strip()])

    @staticmethod
    def _story_text_units(text: str, language_code: Optional[str]) -> int:
        """Return a language-safe lightweight content-size measure.

        Existing whitespace word counts remain unchanged for space-delimited
        languages. Japanese uses visible Japanese/alphanumeric characters
        because ``split()`` collapses a whole Japanese sentence into one token.
        """
        language = str(language_code or "en").strip().lower().replace("_", "-").split("-", 1)[0]
        cleaned = str(text or "").strip()
        if language == "ja":
            return len(re.findall(r'[\u3040-\u30ff\u3400-\u9fff々〆ヵヶA-Za-z0-9]', cleaned))
        return len(cleaned.split())

    @staticmethod
    def _first_page_minimum_units(child_age: int, language_code: Optional[str]) -> tuple[int, str]:
        """Return the existing Page-1 floor, adapted only where words are unsafe.

        English and other space-delimited languages retain the exact historical
        28/38/48-word thresholds. Japanese uses a conservative character floor
        while sentence-count validation continues to provide the stronger shape
        guard.
        """
        language = str(language_code or "en").strip().lower().replace("_", "-").split("-", 1)[0]
        if language == "ja":
            return (45 if child_age <= 5 else 60 if child_age <= 8 else 75), "chars"
        return (28 if child_age <= 5 else 38 if child_age <= 8 else 48), "words"

    def _sanitize_generated_page_text(self, text: Any) -> str:
        """Remove model-only page labels and repair obvious formatting splits.

        This is presentation cleanup only. It does not rewrite story content,
        alter page order, change narration ownership, or affect Page-1-first
        generation.
        """
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""

        number_words = (
            "one|two|three|four|five|six|seven|eight|nine|ten|"
            "eleven|twelve"
        )
        cleaned = re.sub(
            rf"^\s*page\s+(?:\d+|{number_words})\s*(?::|[-–—.]\s*)?",
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        ).lstrip()

        # Gemini can occasionally leave JSON-style escaped quote markers in
        # the decoded story string (for example: \\"You may go...\\").
        # Story pages are prose, so remove only the stray escape before a
        # double quote. This keeps the visible text and narration text clean.
        cleaned = cleaned.replace('\\"', '"')

        # Story prose is plain text. Remove common inline Markdown formatting
        # markers if the model leaks them despite the prompt. This does not
        # rewrite wording or story meaning; it only removes presentation markup.
        cleaned = re.sub(r"\*{1,3}([^*\n]+?)\*{1,3}", r"\1", cleaned)
        cleaned = re.sub(r"_{1,2}([^_\n]+?)_{1,2}", r"\1", cleaned)
        cleaned = re.sub(r"`{1,3}([^`\n]+?)`{1,3}", r"\1", cleaned)
        cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", cleaned)
        cleaned = re.sub(r"(?m)^\s*[-+*]\s+(?=\S)", "", cleaned)

        # Remove accidental single line wrapping inside paragraphs.
        cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)

        # Repair accidental paragraph breaks inside names/titles.
        cleaned = re.sub(
            r"\b(Mr|Mrs|Ms|Miss|Dr|Prof|St)\.\s*\n\s*\n\s*"
            r"(?=[A-ZÀ-ÖØ-Þ])",
            r"\1. ",
            cleaned,
        )

        # Repair a paragraph break after a comma when the sentence continues.
        cleaned = re.sub(
            r",\s*\n\s*\n\s*(?=[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ])",
            ", ",
            cleaned,
        )

        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n\s*\n\s*\n+", "\n\n", cleaned)
        return cleaned.strip()

    def _sanitize_generated_pages(self, pages: Any) -> list[str]:
        if not isinstance(pages, list):
            return []
        return [
            cleaned
            for page in pages
            if (cleaned := self._sanitize_generated_page_text(page))
        ]

    def _valid_generated_pages(self, pages: Any, expected_count: int, request: Optional[GenerateStoryRequest] = None) -> list[str]:
        """Normalize and validate a generated continuation page batch.

        Phase 11 quality guard: pages 2+ must be real story pages, not
        one-sentence captions. If Gemini returns thin pages, reject the batch so
        the existing retry path can ask for a smaller batch and regenerate.
        """
        if not isinstance(pages, list):
            return []
        processed = self._sanitize_generated_pages(postprocess_story_pages(pages))[:expected_count]
        valid_pages: list[str] = []
        for index, page in enumerate(processed, start=1):
            text = str(page or "").strip()
            language = (request.storyLanguageCode if request is not None else "en")
            word_count = self._story_text_units(text, language)
            sentence_count = self._count_story_sentences(text)
            paragraph_count = len([p for p in re.split(r'\n\s*\n', text) if p.strip()])

            if request is not None and self._is_canon_request(request):
                leak_reason = self._canon_instruction_leak_reason(text)
                if leak_reason:
                    print(
                        f"[PERF] canon_generated_page_rejected index={index} "
                        f"reason={leak_reason}"
                    )
                    continue

            # Keep the legacy floor for standard/Canon stories. Living World
            # has a stronger age-aware floor so synopsis-sized pages cannot pass
            # while still allowing natural variation below the preferred target.
            base_language = str(language or "en").strip().lower().replace("_", "-").split("-", 1)[0]
            min_words = 80 if base_language == "ja" else 50
            min_sentences = 3
            if request is not None and self._is_folk_adventure_request(request):
                child_age = self._safe_child_age(request.age)
                if child_age <= 4:
                    min_words, min_sentences = 45, 3
                elif child_age <= 6:
                    min_words, min_sentences = 60, 4
                elif child_age <= 8:
                    min_words, min_sentences = 80, 4
                elif child_age <= 10:
                    min_words, min_sentences = 90, 4
                else:
                    min_words, min_sentences = 95, 4

            if (
                word_count >= min_words
                and sentence_count >= min_sentences
            ):
                valid_pages.append(text)
            else:
                print(
                    f"[PERF] generated_page_rejected index={index} "
                    f"words={word_count} sentences={sentence_count} "
                    f"paragraphs={paragraph_count}"
                )
        return valid_pages

    @staticmethod
    def _normalise_boundary_overlap_text(text: str) -> str:
        """Normalise story text only for exact page-boundary overlap comparison."""
        cleaned = str(text or "").strip().casefold()
        cleaned = (
            cleaned
            .replace("“", '"').replace("”", '"').replace("„", '"')
            .replace("«", '"').replace("»", '"')
            .replace("’", "'").replace("‘", "'")
            .replace("–", "-").replace("—", "-")
        )
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip()

    @staticmethod
    def _story_sentence_spans(text: str) -> list[tuple[int, int, str]]:
        """Return multilingual sentence spans while preserving source offsets."""
        source = str(text or "")
        if not source.strip():
            return []
        spans: list[tuple[int, int, str]] = []
        pattern = re.compile(r'.+?(?:[.!?؟。！？]+(?:["”’»\)]*)|$)', flags=re.DOTALL)
        for match in pattern.finditer(source):
            sentence = match.group(0).strip()
            if sentence:
                spans.append((match.start(), match.end(), sentence))
        return spans

    def _remove_page_boundary_duplicate(
        self,
        previous_page: str,
        new_page: str,
        language_code: Optional[str] = "en",
    ) -> tuple[str, bool, str]:
        """Remove only a confirmed repeated prefix copied from the prior page."""
        previous = self._sanitize_generated_page_text(previous_page)
        candidate = self._sanitize_generated_page_text(new_page)
        if not previous or not candidate:
            return candidate, False, "empty_boundary"

        prev_spans = self._story_sentence_spans(previous)
        new_spans = self._story_sentence_spans(candidate)
        if not prev_spans or not new_spans:
            return candidate, False, "no_sentence_spans"

        max_sentences = min(4, len(prev_spans), len(new_spans))
        for count in range(max_sentences, 0, -1):
            prev_chunk = " ".join(span[2] for span in prev_spans[-count:])
            new_chunk = " ".join(span[2] for span in new_spans[:count])
            prev_norm = self._normalise_boundary_overlap_text(prev_chunk)
            new_norm = self._normalise_boundary_overlap_text(new_chunk)
            if not prev_norm or prev_norm != new_norm:
                continue

            base_lang = str(language_code or "en").strip().lower().replace("_", "-").split("-", 1)[0]
            if base_lang == "ja":
                substantial = len(re.findall(r'[\u3040-\u30ff\u3400-\u9fff々〆ヵヶA-Za-z0-9]', new_norm)) >= 20
            else:
                substantial = len(new_norm.split()) >= 8
            if not substantial:
                continue

            cut_at = new_spans[count - 1][1]
            cleaned = self._sanitize_generated_page_text(candidate[cut_at:].lstrip(" \t\r\n"))
            if not cleaned:
                return candidate, False, "overlap_would_empty_page"
            return cleaned, True, f"exact_{count}_sentence_boundary_overlap"

        prev_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", previous) if p.strip()]
        new_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", candidate) if p.strip()]
        if prev_paragraphs and new_paragraphs:
            prev_norm = self._normalise_boundary_overlap_text(prev_paragraphs[-1])
            new_norm = self._normalise_boundary_overlap_text(new_paragraphs[0])
            if prev_norm and prev_norm == new_norm:
                base_lang = str(language_code or "en").strip().lower().replace("_", "-").split("-", 1)[0]
                substantial = (
                    len(re.findall(r'[\u3040-\u30ff\u3400-\u9fff々〆ヵヶA-Za-z0-9]', new_norm)) >= 20
                    if base_lang == "ja"
                    else len(new_norm.split()) >= 8
                )
                if substantial and len(new_paragraphs) > 1:
                    cleaned = self._sanitize_generated_page_text("\n\n".join(new_paragraphs[1:]))
                    if cleaned:
                        return cleaned, True, "exact_paragraph_boundary_overlap"

        return candidate, False, "no_exact_boundary_overlap"

    def _ending_text_without_marker(self, text: str) -> str:
        return re.sub(r"\s*The End\.\s*$", "", str(text or "").strip(), flags=re.IGNORECASE).strip()

    def _ensure_the_end(self, text: str) -> str:
        """Guarantee the final stored page ends with the exact closing marker."""
        body = self._ending_text_without_marker(text)
        if not body:
            return "The End."
        return f"{body}\n\nThe End."

    def _sentence_fingerprints(self, text: str) -> set[str]:
        fingerprints: set[str] = set()
        for sentence in re.split(r'(?<=[.!?])\s+', str(text or "")):
            normalized = re.sub(r"[^a-z0-9 ]+", "", sentence.lower())
            normalized = re.sub(r"\s+", " ", normalized).strip()
            if len(normalized.split()) >= 5:
                fingerprints.add(normalized)
        return fingerprints

    def _validate_final_page(
        self,
        page: str,
        existing_pages: list[str],
        language_code: Optional[str] = "en",
    ) -> tuple[bool, str]:
        """Reject obvious mid-story, repeated, or open-ended final pages.

        Japanese does not delimit words with spaces, so the historical 45-word
        check must use the existing language-safe story-unit counter instead.
        """
        body = self._ending_text_without_marker(page)
        language = str(language_code or "en").strip().lower().replace("_", "-").split("-", 1)[0]
        content_units = self._story_text_units(body, language)
        minimum_units = 80 if language == "ja" else 45
        if content_units < minimum_units:
            return False, "final_page_too_short"
        if self._count_story_sentences(body) < 3:
            return False, "final_page_too_few_sentences"
        if body.rstrip().endswith(("?", "？", "؟")):
            return False, "final_page_ends_with_question"

        lower = body.lower()

        # Reject common AI-style emotional summaries so the focused final-page
        # retry produces a concrete story-specific closing image or callback.
        final_sentences = [
            part.strip()
            for part in re.split(r'(?<=[.!?])\s+', body)
            if part.strip()
        ]
        final_sentence = final_sentences[-1].lower() if final_sentences else ""
        generic_final_patterns = (
            "felt like the bravest",
            "felt very brave",
            "felt so brave",
            "felt very proud",
            "felt so proud",
            "was very proud",
            "was the bravest",
            "was ready for anything",
            "could do anything",
            "would never forget",
            "had learned",
            "realised that",
            "realized that",
            "understood that",
        )
        if any(pattern in final_sentence for pattern in generic_final_patterns):
            return False, "final_sentence_is_generic_emotional_summary"

        unfinished_patterns = (
            "now for ",
            "what would happen next",
            "what happened next",
            "to be continued",
            "this was only the beginning",
            "the adventure was just beginning",
            "another adventure waited",
            "another adventure was waiting",
            "couldn't wait for the next",
            "could not wait for the next",
            "a new mystery",
            "a new clue",
        )
        if any(pattern in lower[-260:] for pattern in unfinished_patterns):
            return False, "final_page_contains_continuation_hook"

        previous_fingerprints: set[str] = set()
        for previous_page in existing_pages:
            previous_fingerprints.update(self._sentence_fingerprints(previous_page))
        final_fingerprints = self._sentence_fingerprints(body)
        if final_fingerprints:
            overlap = len(final_fingerprints & previous_fingerprints) / len(final_fingerprints)
            if overlap >= 0.45:
                return False, f"final_page_repeats_previous_content_{overlap:.2f}"

        return True, "ok"

    async def _review_canon_final_page_semantics(
        self,
        request: GenerateStoryRequest,
        title: str,
        existing_pages: list[str],
        candidate_page: str,
    ) -> tuple[bool, str, list[str], dict[str, bool]]:
        """Verify that a Canon Page 7 completes the recorded folklore ending."""
        if not self.model:
            return True, "review_skipped_no_model", [], {}

        contract = self._canon_contract(request)
        story_text = "\n\n".join(
            f"Page {index + 1}: {page}"
            for index, page in enumerate([*existing_pages, candidate_page])
        )
        prompt = f"""Review the final page of a canonical children's folklore retelling.
Return only JSON matching the supplied schema. Use the Canon record as the authority.
Reject an ending that is grammatically finished but does not complete the defining folklore ending.
Do not apply normal PillowTales requirements for a child-led solution, parent theme, or selected moral.

EXACT STORY TITLE:
{title}

CANON RECORD:
{json.dumps(contract, ensure_ascii=False, separators=(',', ':'))}

FULL GENERATED STORY:
{story_text}

REVIEW RULES:
- canonical_ending_complete: the defining ending in ending_rules and required events is fully shown, not summarised away or stopped early.
- required_final_events_present: all events needed to complete the recorded ending appear in substance.
- required_event_order_preserved: the defining events remain in their recorded order.
- canonical_characters_preserved: legendary characters keep their identities, roles, motivations, and outcomes.
- no_invented_resolution: no new object, helper, villain, quest, moral, or substitute solution resolves the legend.
- child_does_not_change_outcome: the child remains outside the legend and does not enter, observe, accompany, speak to, assist, warn, replace, or alter any canonical character, event, or result.
- no_unfinished_canon_event: no recorded event remains pending and there is no sequel hook.
- satisfying_canonical_close: the story ends naturally inside the canonical narrative with no listener frame, parent exchange, epilogue or commentary.
- required_changes: give 1-6 precise imperative repairs using only the Canon record and material already established.
"""
        try:
            response = await asyncio.to_thread(
                self._generate_content_sync,
                prompt,
                self._canon_ending_review_response_schema(),
                1600,
            )
            response_text = getattr(response, "text", None)
            if not response_text or not isinstance(response_text, str):
                return False, "canon_semantic_review_empty_response", ["Complete the recorded canonical ending without inventing a replacement resolution."], {}
            result = self._clean_json_response(response_text)
            checks = (
                "canonical_ending_complete",
                "required_final_events_present",
                "required_event_order_preserved",
                "canonical_characters_preserved",
                "no_invented_resolution",
                "child_does_not_change_outcome",
                "no_unfinished_canon_event",
                "satisfying_canonical_close",
            )
            check_results = {key: result.get(key) is True for key in checks}
            failed = [key for key, passed in check_results.items() if not passed]
            required_changes = [
                str(item).strip()
                for item in (result.get("required_changes") or [])
                if str(item).strip()
            ][:6]
            if failed:
                if not required_changes:
                    required_changes = [
                        "Complete every final Canon event in the recorded order.",
                        "Remove any invented resolution or child-led change to the outcome.",
                        "Close only after the defining folklore ending is fully shown.",
                    ]
                reason = str(result.get("reason") or "").strip()
                return False, f"canon_semantic_review_failed:{','.join(failed)}:{reason}"[:700], required_changes, check_results
            return True, "ok", [], check_results
        except Exception as exc:
            # As with the normal reviewer, reviewer availability must not strand
            # a locally complete Page 7. Log the loss of semantic assurance.
            print(f"[PERF] canon_final_page_semantic_review_unavailable error={str(exc)[:300]}")
            return True, "canon_semantic_review_unavailable", [], {}

    async def _canon_can_finish_on_current_page(
        self,
        request: GenerateStoryRequest,
        title: str,
        pages: list[str],
    ) -> tuple[bool, str]:
        """Allow Canon to finish naturally on Page 6 when the source is complete.

        This is background-only and does not change Page-1-first generation.
        The existing Canon semantic reviewer remains the authority. We only
        shorten the provisional seven-page target when every critical Canon
        completion check passes.
        """
        if not self._is_canon_request(request) or len(pages) < 2:
            return False, "not_applicable"

        semantic_valid, semantic_reason, _, check_results = await self._review_canon_final_page_semantics(
            request=request,
            title=title,
            existing_pages=pages[:-1],
            candidate_page=pages[-1],
        )
        critical_checks = (
            "canonical_ending_complete",
            "required_final_events_present",
            "required_event_order_preserved",
            "canonical_characters_preserved",
            "no_invented_resolution",
            "child_does_not_change_outcome",
            "no_unfinished_canon_event",
            "satisfying_canonical_close",
        )
        critical_pass = bool(check_results) and all(
            check_results.get(key) is True for key in critical_checks
        )
        return bool(semantic_valid and critical_pass), semantic_reason

    async def _review_final_page_semantics(
        self,
        request: GenerateStoryRequest,
        title: str,
        existing_pages: list[str],
        candidate_page: str,
    ) -> tuple[bool, str, list[str], dict[str, bool]]:
        """Review whether Page 7 genuinely completes the story."""
        if not self.model:
            return True, "review_skipped_no_model", [], {}

        moral_required = self._moral_requested(request) and not self._is_folk_adventure_request(request)
        story_text = "\n\n".join(
            f"Page {index + 1}: {page}" for index, page in enumerate([*existing_pages, candidate_page])
        )
        spine = self._story_spine_block(request=request, title=title, first_page=(existing_pages or [""])[0])
        moral_rule = (
            "- moral_visible_through_action: the requested moral is demonstrated by a meaningful choice or consequence, not merely stated."
            if moral_required else
            "- No moral was requested. Do not assess, request or repair a moral."
        )
        mode_rules = (
            """LIVING WORLD ENDING REVIEW:
- Page 6 should contain the decisive action/climax or make the resolution inevitable.
- Page 7 may complete that already-established climax, but it must also provide consequence, relief, callback and safe settling.
- Do NOT demand a new solution if Pages 1-6 already resolved the main problem.
- Reject Page 7 if it invents a new cause, unrelated clue, unestablished solution method, new task, or sequel hook."""
            if self._is_folk_adventure_request(request) else
            """STANDARD BEDTIME STORY ENDING REVIEW:
- Judge this as a complete seven-page children's story, not merely as a locally valid final paragraph.
- Page 7 must answer or fulfil the wish, question, promise, relationship need, or problem introduced on Page 1.
- The decisive result must be earned by the child's earlier actions, choices, relationships, clues, habits, or established story rules.
- A generic celebration, generic praise, sudden gift, unexplained magical fix, or interchangeable bedtime paragraph is not a satisfying ending.
- If the ending could be swapped into another story with only names changed, reject it.
- The final callback should come from something established earlier and create recognition rather than summarising the lesson."""
        )
        prompt = f"""Review the ending of this children's bedtime story.
Return only JSON matching the supplied schema. Be demanding about story quality but do not invent extra plot when repairing.

{spine}

FULL STORY INCLUDING CANDIDATE FINAL PAGE:
{story_text}

{mode_rules}

REVIEW RULES:
- resolves_opening_promise: the Page 1 wish, question, promise, relationship need, or problem is genuinely answered or fulfilled by the end.
- resolves_main_problem: the central external problem is actually completed, not merely beginning to complete.
{moral_rule}
- emotional_payoff_complete: the promised relationship or emotional change is shown through action or dialogue.
- callback_earned: at least one earlier detail returns with purpose.
- no_new_plot: Page 7 adds no new character, task, mystery, object, place, rule, or sequel hook.
- ending_feels_earned: the ending depends on specific events, choices, relationships, clues, habits, or rules from Pages 2-6.
- satisfying_ending: the result feels like the natural destination of Pages 1-6 and includes a brief settled afterglow, not simply a stopped scene.
- required_changes: give 1-6 short imperative repairs using only established material from Pages 1-6.
"""
        try:
            response = await asyncio.to_thread(
                self._generate_content_sync,
                prompt,
                self._ending_review_response_schema(include_moral=moral_required),
                1400,
            )
            response_text = getattr(response, "text", None)
            if not response_text or not isinstance(response_text, str):
                return False, "semantic_review_empty_response", ["Complete the original opening promise and main problem."], {}
            result = self._clean_json_response(response_text)
            checks = [
                "resolves_opening_promise", "resolves_main_problem", "emotional_payoff_complete",
                "callback_earned", "no_new_plot", "ending_feels_earned", "satisfying_ending",
            ]
            if moral_required:
                checks.append("moral_visible_through_action")
            check_results = {key: result.get(key) is True for key in checks}
            failed = [key for key, passed in check_results.items() if not passed]
            required_changes = [str(item).strip() for item in (result.get("required_changes") or []) if str(item).strip()][:6]
            if failed:
                reason = str(result.get("reason") or "").strip()
                if not required_changes:
                    required_changes = [
                        "Complete the original opening promise and main problem on Page 7.",
                        "Show the emotional result through action or dialogue.",
                        "End with an established story-specific callback and a settled final image.",
                    ]
                return False, f"semantic_review_failed:{','.join(failed)}:{reason}"[:700], required_changes, check_results
            return True, "ok", [], check_results
        except Exception as exc:
            print(f"[PERF] final_page_semantic_review_skipped error={str(exc)[:300]}")
            return True, "semantic_review_unavailable", [], {}

    def _final_page_repair_block(
        self,
        rejection_reason: Optional[str],
        required_changes: Optional[list[str]],
        repair_attempt: int,
        completion_first: bool = False,
    ) -> str:
        """Build targeted Page 7 repair instructions from the reviewer output."""
        changes = required_changes or [
            "Complete the original opening promise and main problem.",
            "Show the emotional payoff through action or dialogue.",
            "Use an earlier story detail as the final callback.",
        ]
        change_lines = "\n".join(f"- {item}" for item in changes[:6])
        priority = """
COMPLETION-FIRST FINAL REPAIR:
- This is the last recovery pass. A complete, coherent ending is mandatory.
- Do not stop while an egg is cracking, a door is opening, a character is arriving, or a solution is merely beginning.
- Show the promised outcome fully happening on this page.
- Include the relationship or emotional result promised by Page 1.
- Use only established characters, objects, rules, actions, and locations.
""" if completion_first else ""
        return f"""
TARGETED FINAL PAGE REPAIR — ATTEMPT {repair_attempt}:
The ending reviewer rejected the previous Page 7 for this exact reason:
{rejection_reason or 'The ending did not fully complete the story.'}

REQUIRED CHANGES — APPLY EVERY ONE:
{change_lines}

- Rewrite Page 7 from scratch; do not merely edit its final sentence.
- Resolve the exact promise and problem established by Pages 1-6.
- The decisive solution must come from the established protagonist or from consequences already earned in the story.
- Show the completed result, then a brief emotional afterglow, then a safe settled final image.
- Do not introduce a new method, character, object, clue, sound, task, place, or surprise.
- Do not end at the instant something begins to happen. Show what happens and what it means to the existing characters.
- End with one concrete callback from Pages 1-6 followed by exactly: The End.
{priority}
"""


    async def _generate_remaining_pages_batch(
        self,
        request: GenerateStoryRequest,
        companion: Optional[dict],
        title: str,
        working_pages: list[str],
        batch_count: int,
    ) -> list[str]:
        """Generate and validate a continuation batch.

        Page 7 uses reviewer-guided repair. Each retry receives the exact failed
        criteria and required changes from the previous semantic review. A final
        completion-first repair may be accepted only when all critical ending
        checks pass; this prevents a quality preference from leaving the story
        permanently at six pages.
        """
        next_page_number = len(working_pages) + 1
        intended_final_page = self._intended_page_count(request)
        is_final_page_batch = batch_count == 1 and next_page_number == intended_final_page
        max_attempts = (
            5
            if is_final_page_batch
            else (3 if self._is_canon_request(request) else BACKGROUND_PAGE_MAX_ATTEMPTS)
        )
        last_error: Optional[str] = None
        last_required_changes: list[str] = []

        for generation_attempt in range(1, max_attempts + 1):
            prompt = self._build_remaining_pages_prompt(
                request=request,
                companion=companion,
                title=title,
                existing_pages=working_pages,
                remaining_page_count=batch_count,
                next_page_number=next_page_number,
            )
            if (
                self._is_canon_request(request)
                and not is_final_page_batch
                and generation_attempt > 1
            ):
                prompt += f"""
CANON CONTINUATION PAGE REPAIR — ATTEMPT {generation_attempt}:
- The previous continuation candidate was rejected as too thin or incomplete.
- Rewrite Page {next_page_number} as a complete story page.
- Use 90-150 words, 5-7 read-aloud sentences, and exactly 2 short paragraphs.
- Do not return fewer than 80 words.
- Expand the protected Canon material assigned to Page {next_page_number} with concrete action, brief dialogue, reactions and place detail.
- If the Canon event budget requires more than one ordered event on this page, progress through those events naturally rather than stretching one event.
- Do not advance merely to add length, and do not skip or reorder required events.
- Do not recap Page {next_page_number - 1}.
- Return only the required JSON.
"""

            if is_final_page_batch and generation_attempt > 1:
                if self._is_canon_request(request):
                    prompt += f"""
CANON FINAL PAGE REPAIR — ATTEMPT {generation_attempt}:
- The previous page failed Canon completeness review: {last_error or 'unknown failure'}.
- Required repairs from the reviewer: {json.dumps(last_required_changes, ensure_ascii=False)}
- Rewrite the final page from scratch using only the protected Canon record and events already established in Pages 1-6.
- Explicitly complete every reviewer-listed missing Canon event in the recorded order.
- Do not repeat already-completed earlier events simply to fill space.
- Do not invent a new resolution, moral, object, helper, callback, or child-led solution.
- If several protected closing events remain, use concise scene transitions so they ALL fit naturally on this final page.
- Fully reach the protected required ending, add no listener frame, parent exchange, epilogue or commentary, and end exactly with The End.
"""
                else:
                    prompt += self._final_page_repair_block(
                        rejection_reason=last_error,
                        required_changes=last_required_changes,
                        repair_attempt=generation_attempt,
                        completion_first=(generation_attempt == max_attempts),
                    )

            print(
                f"[PERF] remaining_pages_batch prompt chars={len(prompt)} "
                f"next_page={next_page_number} count={batch_count} attempt={generation_attempt}"
            )

            t_gemini = time.time()
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._generate_content_sync,
                        prompt,
                        self._story_response_schema(batch_count, include_title=False),
                    ),
                    timeout=BACKGROUND_PAGE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - t_gemini
                last_error = (
                    f"Gemini continuation timed out after "
                    f"{BACKGROUND_PAGE_TIMEOUT_SECONDS}s on Page {next_page_number}"
                )
                last_required_changes = []
                print(
                    f"[PERF] remaining_pages_batch TIMEOUT after={elapsed:.2f}s "
                    f"next_page={next_page_number} count={batch_count} "
                    f"attempt={generation_attempt}/{max_attempts}"
                )
                continue

            print(
                f"[PERF] remaining_pages_batch Gemini took {time.time() - t_gemini:.2f}s "
                f"next_page={next_page_number} count={batch_count} attempt={generation_attempt}"
            )

            response_text = getattr(response, 'text', None)
            if not response_text or not isinstance(response_text, str):
                last_error = "empty response text"
                last_required_changes = ["Return one complete Page 7 as valid JSON."]
                continue

            try:
                story_data = self._clean_json_response(response_text)
                if not isinstance(story_data, dict) or 'pages' not in story_data:
                    raise ValueError("Invalid remaining-pages batch format returned by AI")
                batch_pages = self._valid_generated_pages(story_data.get('pages', []), batch_count, request=request)
            except Exception as parse_exc:
                print(
                    f"[PERF] remaining_pages_batch parse failed "
                    f"next_page={next_page_number} count={batch_count} "
                    f"attempt={generation_attempt}: {parse_exc}"
                )
                batch_pages = self._salvage_pages_from_response_text(response_text, batch_count)

            if len(batch_pages) < batch_count:
                last_error = (
                    f"Remaining generation produced only {len(batch_pages)} "
                    f"of {batch_count} pages in batch"
                )
                last_required_changes = ["Return one complete final page with 5-7 sentences and two short paragraphs."]
                continue

            sanitized = self._sanitize_generated_pages(batch_pages[:batch_count])

            # Deterministic page-boundary de-duplication.
            # Remove only exact confirmed overlap copied from the end of the
            # previous page into the start of the new page.
            boundary_cleaned: list[str] = []
            previous_for_boundary = working_pages[-1] if working_pages else ""
            boundary_rejected = False
            for offset, candidate_page in enumerate(sanitized):
                cleaned_page, changed, overlap_reason = self._remove_page_boundary_duplicate(
                    previous_for_boundary,
                    candidate_page,
                    request.storyLanguageCode,
                )
                page_number = next_page_number + offset

                if changed:
                    print(
                        f"[PERF] page_boundary_duplicate_removed "
                        f"page={page_number} reason={overlap_reason}"
                    )
                    revalidated = self._valid_generated_pages(
                        [cleaned_page],
                        1,
                        request=request,
                    )
                    if len(revalidated) != 1:
                        last_error = (
                            f"Page {page_number} became too thin after removing "
                            f"repeated content from Page {page_number - 1}"
                        )
                        last_required_changes = [
                            f"Write Page {page_number} with new story content only.",
                            f"Do not repeat or recap Page {page_number - 1}.",
                        ]
                        print(
                            f"[PERF] page_boundary_duplicate_rejected_after_cleanup "
                            f"page={page_number} reason={overlap_reason}"
                        )
                        boundary_rejected = True
                        break
                    cleaned_page = revalidated[0]

                boundary_cleaned.append(cleaned_page)
                previous_for_boundary = cleaned_page

            if boundary_rejected:
                continue

            sanitized = boundary_cleaned
            if is_final_page_batch:
                is_canon = self._is_canon_request(request)
                valid, reason = self._validate_final_page(sanitized[0], working_pages, request.storyLanguageCode)
                repetition_only = reason.startswith("final_page_repeats_previous_content_")
                if not valid and not (is_canon and repetition_only):
                    last_error = reason
                    last_required_changes = [
                        "Write a complete final page rather than a caption or repeated scene.",
                        "Finish the original problem and end without a question or sequel hook.",
                    ]
                    print(
                        f"[PERF] final_page_rejected attempt={generation_attempt} "
                        f"story_title={title!r} reason={reason}"
                    )
                    continue

                if is_canon and repetition_only:
                    # Canon closings can legitimately mention the immediately
                    # preceding defining event, so do not let the generic local
                    # overlap gate prevent the Canon semantic reviewer from
                    # inspecting the complete story. A repeated candidate is
                    # still rejected below unless the semantic review passes and
                    # the page is sufficiently distinct to serve as a closing.
                    last_error = reason
                    print(
                        f"[PERF] canon_final_page_repetition_sent_to_semantic_review "
                        f"attempt={generation_attempt} story_title={title!r} reason={reason}"
                    )

                if is_canon:
                    semantic_valid, semantic_reason, required_changes, check_results = await self._review_canon_final_page_semantics(
                        request=request,
                        title=title,
                        existing_pages=working_pages,
                        candidate_page=sanitized[0],
                    )
                else:
                    semantic_valid, semantic_reason, required_changes, check_results = await self._review_final_page_semantics(
                        request=request,
                        title=title,
                        existing_pages=working_pages,
                        candidate_page=sanitized[0],
                    )
                if not semantic_valid:
                    last_error = semantic_reason
                    last_required_changes = required_changes
                    print(
                        f"[PERF] final_page_semantic_rejected attempt={generation_attempt} "
                        f"story_title={title!r} reason={semantic_reason} "
                        f"required_changes={required_changes}"
                    )

                    # On the final completion-first pass, accept only when the
                    # essential completion criteria pass. Secondary polish may
                    # never strand a child at six pages.
                    critical_checks = (
                        (
                            "canonical_ending_complete",
                            "required_final_events_present",
                            "no_invented_resolution",
                            "child_does_not_change_outcome",
                            "no_unfinished_canon_event",
                        )
                        if self._is_canon_request(request)
                        else (
                            "resolves_opening_promise",
                            "resolves_main_problem",
                            "emotional_payoff_complete",
                            "no_new_plot",
                            "satisfying_ending",
                        )
                    )
                    critical_pass = bool(check_results) and all(
                        check_results.get(key) is True for key in critical_checks
                    )
                    if generation_attempt < max_attempts or not critical_pass:
                        continue
                    print(
                        f"[PERF] final_page_completion_validated_with_secondary_warnings "
                        f"story_title={title!r} checks={check_results}"
                    )

                if is_canon and repetition_only:
                    # Even if the full Canon story is semantically complete, do
                    # not publish a Page 7 that is substantially a duplicate of
                    # Page 6. Keep trying for a distinct aftermath/bedtime close.
                    last_required_changes = [
                        "Do not retell or paraphrase Page 6.",
                        "Begin after the last completed canonical event.",
                        "Complete only any remaining canonical aftermath, then return briefly to the external bedtime frame.",
                    ]
                    print(
                        f"[PERF] canon_final_page_repetition_rejected_after_semantic_review "
                        f"attempt={generation_attempt} story_title={title!r} reason={reason}"
                    )
                    continue

                sanitized[0] = self._ensure_the_end(sanitized[0])
                print(
                    f"[PERF] final_page_accepted attempt={generation_attempt} "
                    f"story_title={title!r}"
                )

            return sanitized

        if is_final_page_batch and self._is_canon_request(request):
            # Do not manufacture a meta-story Page 7. If Canon was already
            # complete on Page 6, complete_story_background now detects that
            # before Page 7 is requested. If genuine Canon still remained and
            # all Page 7 attempts failed, preserve the partial story for retry
            # rather than publishing commentary such as "the tale was complete".
            print(
                f"[PERF] canon_final_page_no_meta_fallback story_title={title!r} "
                f"previous_error={last_error!r}"
            )

        raise ValueError(
            f"Remaining generation failed after {max_attempts} attempt(s): "
            f"{last_error or 'unknown error'}"
        )

    def _publish_partial_story_pages(
        self,
        story_id: str,
        user_id: str,
        working_pages: list[str],
        expected_pages: int,
        generation_error: Optional[str] = None,
    ) -> None:
        """Persist currently usable pages without marking the story failed."""
        partial_text = '\n\n'.join(working_pages)
        t_partial_update = time.time()
        print(
            f"[PERF] story_update_partial START story_id={story_id} "
            f"pages={len(working_pages)}/{expected_pages}"
        )
        payload = {
            'pages': working_pages,
            'full_text': partial_text,
            'generation_status': 'partial',
            'expected_pages': expected_pages,
            'generation_error': generation_error[:500] if generation_error else None,
        }
        self.story_repo.update(story_id, user_id, payload)
        print(
            f"[PERF] story_update_partial DONE story_id={story_id} "
            f"total={time.time() - t_partial_update:.2f}s"
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
        if child_age <= 6:
            return """AGE-SPECIFIC PAGE 1 RULES — 6:
- Write like an early-reader bedtime story, not a children's novel.
- Most sentences should be 5-10 words.
- Use everyday spoken vocabulary and concrete actions.
- Use one clear place, one clear trigger, one clear problem, and one next step.
- Use no poetic narration, no abstract lesson language, and no crowded magical setup.
- Dialogue should be short and plain.
- If choosing between beautiful wording and easy wording, choose easy wording."""
        if child_age <= 8:
            return """AGE-SPECIFIC PAGE 1 RULES — 7-8:
- Use confident child-friendly adventure language.
- Most sentences should be 8-14 words.
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

    def _canon_age_storytelling_block(self, age: Any) -> str:
        """Canon-specific age calibration: Canon controls WHAT; age controls HOW."""
        child_age = self._safe_child_age(age)

        if child_age <= 4:
            level = """CANON READING/READ-ALOUD LEVEL — AGE 0-4:
- Use very short, concrete sentences and familiar words.
- Keep cause and effect explicit and easy to follow.
- Use brief natural dialogue only where it helps understanding.
- Describe frightening, violent, cruel or adult material very gently and without graphic detail."""
        elif child_age <= 6:
            level = """CANON READING/READ-ALOUD LEVEL — AGE 5-6:
- Use Oxford-inspired early-reader clarity: mostly short sentences, one clear idea at a time, familiar vocabulary and concrete actions.
- Let important Canon moments happen as simple scenes with short dialogue and visible reactions.
- Keep motivations clear rather than abstract.
- Soften frightening, violent, cruel, sexual or otherwise adult presentation, but preserve the event and its cause when part of Canon."""
        elif child_age <= 8:
            level = """CANON READING/READ-ALOUD LEVEL — AGE 7-8:
- Use Oxford-inspired confident early chapter-book storytelling.
- Use varied but clear sentences, richer child-friendly vocabulary, regular characterful dialogue and connected scenes.
- Let motives and feelings emerge through actions and dialogue rather than narrator summary.
- Important Canon moments should be dramatized as scenes, not reduced to plot-summary sentences."""
        elif child_age <= 10:
            level = """CANON READING/READ-ALOUD LEVEL — AGE 9-10:
- Use Oxford-inspired middle-grade clarity with richer vocabulary, varied sentence structure, stronger dialogue, motives, anticipation and consequence.
- Allow the reader to infer some feelings and intentions.
- Give defining Canon scenes enough atmosphere and emotional weight to feel lived rather than reported."""
        else:
            level = """CANON READING/READ-ALOUD LEVEL — AGE 11-12:
- Use the strongest fluent-child level in the PillowTales Oxford-inspired scale.
- Use nuanced but clear vocabulary, varied sentence structure, layered dialogue, implication, motivation, atmosphere and emotional consequence.
- Do not talk down to the reader or over-explain feelings.
- Let major Canon scenes breathe with anticipation, reaction and subtext while remaining children's fiction and bedtime-safe."""

        return level + """

CANON AGE-ADAPTATION LAW — NON-NEGOTIABLE:
- CANON determines WHAT happens: required characters, relationships, events, event order, transformations, consequences and ending.
- AGE SAFETY determines HOW disturbing or adult material is expressed.
- READING LEVEL determines HOW sophisticated the prose, dialogue, inference, description and scene depth may be.
- Age adaptation may soften graphic violence, cruelty, horror, sexual intent, adult implications, punishment, death, pregnancy, childbirth or frightening imagery.
- Age adaptation MUST NOT delete a required event, relationship, parentage fact, marriage, birth, death, betrayal, transformation, consequence, cause, rescue, reunion or ending when necessary to the authentic story.
- If an adult or violent Canon event is essential, state it simply and safely rather than replacing it with a different event.
- Never remove marriage, parentage, pregnancy or birth if those facts explain who characters are or why later events happen; tell them without sexual detail.
- Do not sanitise so aggressively that the story stops making causal sense.

CANON STORYTELLING LAW — DRAMATISE, DO NOT INVENT:
- A faithful retelling is still a STORY, not a synopsis, timeline or encyclopedia entry.
- Dramatise authentic events through dialogue, reaction, body language, atmosphere, sensory detail, anticipation and emotional consequence.
- Never invent a new event, villain, helper, object, quest, power, solution or motivation merely to create drama.
- Prefer showing a protected event happening in-scene over summarising it as 'they decided', 'he failed', 'she rescued him', or 'the truth was revealed' when that event deserves narrative space.
- Dialogue may expand a Canon moment only when it preserves the recorded relationship, motivation, fact and outcome.
- For older children, increase scene depth and inference rather than simply using harder vocabulary.
- For younger children, simplify the same authentic scene rather than deleting it.
"""

    def _build_canon_first_page_prompt(self, request: GenerateStoryRequest, companion: Optional[dict]) -> str:
        blocks = self._language_and_character_blocks(request, companion)
        contract = self._canon_contract(request)
        title = contract.get('title') or 'Original Folk Story'
        child_age = self._safe_child_age(request.age)
        max_chars = 560 if child_age <= 6 else 700
        return f"""Write Page 1 only of a faithful canonical folklore retelling. Return only final JSON.

LANGUAGE:
- Write only in {blocks['language_name']}.
- {self._first_page_language_style_block(request.storyLanguageCode)}

LISTENER AGE ONLY:
- Age: {request.age}
- The listener's name is intentionally not supplied to Canon generation.
- Never add or address the listener in the canonical prose.

{self._story_world_prompt_block(request)}

PAGE 1 CANON JOB:
- Use the exact canon title: {title}
- Begin directly inside the first required canonical scene.
- Do not use a random PillowTales opening seed or external bedtime frame.
- Do not invent a trigger, quest, object, helper, joke, mystery, moral or motivation.
- Do not mention, address, describe or refer to the listening child anywhere in the page.
- Only canonical characters may appear or participate.
- Preserve the first required event and its place in the sequence.
- Do not solve or skip ahead.

AGE, STORYTELLING AND SAFETY:
{self._canon_age_storytelling_block(request.age)}

{self._natural_name_pronoun_rules(protect_canon_names=True)}
- Maximum {max_chars} characters.
- 4-6 read-aloud sentences in 1-2 short paragraphs.
- Clear, warm and bedtime-safe, but factually faithful.

JSON ONLY:
{{"title":{json.dumps(str(title), ensure_ascii=False)},"pages":["page 1 text"]}}
- The returned title must match exactly.
"""

    def _canon_event_budget_block(self, request: GenerateStoryRequest, next_page_number: int) -> str:
        """Allocate ordered Canon events across the seven-page maximum."""
        contract = self._canon_contract(request)
        events = contract.get("required_events") or []
        if not isinstance(events, list) or not events:
            return ""
        clean_events = [str(event).strip() for event in events if str(event).strip()]
        if not clean_events:
            return ""
        total = len(clean_events)
        progress_fraction = {2: 0.22, 3: 0.38, 4: 0.54, 5: 0.68, 6: 0.82, 7: 1.00}.get(next_page_number)
        if progress_fraction is None:
            return ""
        target_index = max(1, min(total, int(round(total * progress_fraction))))
        target_event = clean_events[target_index - 1]
        remaining_after_target = total - target_index
        if next_page_number == 7:
            return f"""CANON EVENT COMPLETION BUDGET — FINAL PAGE:
- There are {total} ordered required Canon events.
- Complete every required event not already shown in Pages 1-6, in recorded order.
- Do not repeat completed events merely to mention them again.
- Fully reach the protected required ending before The End.
"""
        return f"""CANON EVENT PACING BUDGET — PAGE {next_page_number}:
- The protected record contains {total} ordered required Canon events.
- By the END of Page {next_page_number}, normally progress through approximately required event {target_index}: {target_event}
- This leaves about {remaining_after_target} required event(s) for later pages.
- This is a pacing floor, not permission to skip, reorder, summarise away, or invent events.
- If already ahead, continue naturally without repetition. If behind, move forward with concise scenes and dialogue rather than stretching one event across pages.
"""

    def _canon_page_pacing_block(self, next_page_number: int) -> str:
        """Pace Canon as scenes without forcing every tale to consume Page 7.

        Canon records remain authoritative for facts, order and ending. The
        provisional seven-page target is a maximum structure, not permission to
        pad a tale after its authentic ending has already been completed.
        """
        rules = {
            2: """CANON PAGE 2 PACING:
- Continue the early canonical sequence and let the first major movement breathe.
- Do not rush through several defining events simply to reach the famous part of the legend.
- Turn the current required event into a scene with concrete action, reaction and brief dialogue where the source permits it.
""",
            3: """CANON PAGE 3 PACING:
- Move through the early-middle canonical events in order.
- Turn required events into scenes with clear action or dialogue rather than a list of things that happened.
- Deepen the current event before advancing; do not consume later events merely to fill the page.
""",
            4: """CANON PAGE 4 PACING:
- Deepen the central canonical relationship, place, change, warning, longing or consequence already recorded.
- Preserve event order and give the current canonical moment enough room to feel like a story scene rather than a summary.
- Do not invent an extra mechanism, explanation or new folklore event to create length.
""",
            5: """CANON PAGE 5 PACING:
- Advance into the late canonical events in order.
- Do not deliberately rush to the ending, but do not withhold or distort a required event merely to reserve material for another page.
- Do not write retrospective legend summaries, moral explanations, cultural commentary or invented explanatory mythology.
- Let the recorded source determine how much canon remains.
""",
            6: """CANON PAGE 6 PACING — NATURAL COMPLETION ALLOWED:
- Show the defining late event or consequence as a full scene, not as a summary.
- If the recorded canonical ending naturally completes on this page, COMPLETE IT fully. Do not withhold part of the authentic ending merely to force Page 7.
- If required canonical events genuinely remain, stop at a natural point and leave only those real events for Page 7.
- Do not add a retrospective legend summary, cultural explanation, moral explanation, invented mechanism or generic bedtime conclusion.
- Do not write 'The End.' on Page 6; the backend will add the marker if semantic Canon review confirms the tale is complete.
""",
            7: """CANON PAGE 7 PACING — ONLY IF CANON REMAINS:
- Page 7 exists only because required canonical material still remained after Page 6.
- Do not retell or paraphrase Page 6.
- Complete only the remaining recorded canonical event(s) and authentic ending.
- Do not explain what the legend means, why it is famous, or what lesson it teaches.
- Do not add an external bedtime frame, listener reaction, parent exchange, epilogue or commentary.
- End naturally inside the canonical narrative, then end exactly with The End.
""",
        }
        return rules.get(next_page_number, "")

    def _build_canon_emergency_closing_page(
        self,
        request: GenerateStoryRequest,
        title: str,
    ) -> str:
        """Deterministic Canon close used only as a last-resort Page 7.

        It adds no listener, parent, narrator or new folklore facts and is safe
        only when semantic review confirms that Pages 1-6 already contain the
        complete Canon ending.
        """
        lang = (request.storyLanguageCode or "en").lower()[:2]
        templates = {
            "en": "The old story had reached its true ending. Nothing more was added, and the tale was complete.",
            "es": "El antiguo cuento había llegado a su verdadero final. No se añadió nada más y la historia quedó completa.",
            "fr": "Le vieux récit avait atteint sa véritable fin. Rien ne fut ajouté, et l'histoire était complète.",
            "de": "Die alte Geschichte hatte ihr wahres Ende erreicht. Nichts wurde hinzugefügt, und die Erzählung war vollständig.",
            "it": "L'antico racconto aveva raggiunto il suo vero finale. Non fu aggiunto altro e la storia era completa.",
        }
        page = templates.get(lang, templates["en"])
        return self._ensure_the_end(page)


    def _build_canon_remaining_pages_prompt(
        self,
        request: GenerateStoryRequest,
        companion: Optional[dict],
        title: str,
        existing_pages: list[str],
        remaining_page_count: int,
        next_page_number: int,
    ) -> str:
        blocks = self._language_and_character_blocks(request, companion)
        existing_pages_text = "\n\n".join(
            f"Page {idx + 1}: {page}" for idx, page in enumerate(existing_pages or [])
        )
        final_page_number = next_page_number + remaining_page_count - 1
        is_final = final_page_number >= self._intended_page_count(request)
        pacing_block = self._canon_page_pacing_block(next_page_number)
        event_budget_block = self._canon_event_budget_block(request, next_page_number)
        final_rule = (
            "Use the recorded canonical ending exactly in substance. Do not substitute a generic PillowTales resolution. "
            "After the canonical ending is fully shown, add no listener frame, parent exchange, epilogue or commentary; end exactly with The End."
            if is_final else
            "Do not end the legend early and do not add a bedtime conclusion yet."
        )
        return f"""Continue a faithful canonical folklore retelling. Return only JSON.

LANGUAGE:
- Write only in {blocks['language_name']}.

{self._story_world_prompt_block(request)}

{event_budget_block}

EXISTING PAGES:
{existing_pages_text}

CANON CONTINUATION JOB:
- Write exactly {remaining_page_count} new page(s): Page {next_page_number} through Page {final_page_number}.
- Continue from the exact next required canon scene or event.
- Preserve required scene order and event order.
- Do not recap, reorder, merge away, replace, or reinterpret defining events.
- Do not create a new child-led mission.
- Do not use parent theme or moral.
- Do not introduce a random magical object, helper, villain, clue, side quest, reward, motivation or solution.
- The listening child must remain completely outside the story. Do not mention, address, describe or refer to the child anywhere; only canonical characters may participate.
- Dialogue may be improved only when it preserves the recorded meaning and motivation.
- Keep cultural names and pronunciation guidance intact.

{self._canon_age_storytelling_block(request.age)}

{self._natural_name_pronoun_rules(protect_canon_names=True)}

- Each page MUST contain exactly 2 short paragraphs and 5-7 read-aloud sentences.
- Each page MUST contain at least 80 words; target 90-150 words.
- A continuation page under 80 words is incomplete. Expand the SAME canonical scene with concrete action, brief dialogue, reactions, physical behaviour and setting detail without adding new canon events.
- SCENE, NOT SUMMARY: let the current required event breathe. Show how it happens instead of compressing it into one or two sentences.
- Never advance into later required events merely to reach the word target.
- Never invent a new causal mechanism, magical explanation, object, weather event, transformation, moral or cultural explanation to create length. Atmosphere may surround a canonical event but must not become the reason the canonical outcome happens.
- If the Canon record says an outcome simply happens (for example something scatters, is lost, is given or is transformed), preserve that outcome without inventing a new folklore explanation for HOW it happened unless the selected source baseline supplies one.

{pacing_block}

FINAL PAGE RULES:
- {final_rule}

OUTPUT JSON:
{{"pages":["new page text"]}}
- Return exactly {remaining_page_count} string(s).
- No notes, markdown or extra keys.
"""

    def _build_canon_first_page_fallback(self, request: GenerateStoryRequest, companion: Optional[dict]) -> Dict[str, Any]:
        """Fail closed if Canon Page 1 cannot be generated safely.

        Canon Page 1 must be real narrative prose. A deterministic fallback built
        from source-control text can expose editorial instructions or create a
        synopsis-sized opening that Pages 2+ then repeat. Do not manufacture a
        Canon page locally. Keep the existing Page-1-first architecture: the
        normal Gemini Page 1 still returns independently and Pages 2+ still load
        in the background; this path is reached only after Page 1 generation has
        already failed or timed out.
        """
        raise HTTPException(
            status_code=503,
            detail='Canonical story generation is temporarily unavailable. Please try again.',
        )


    def _living_world_adapt_original_rules(self, rules: str) -> str:
        """Adapt mature PillowTales quality rules for Living World episodes.

        The original engine remains the quality authority. This adapter removes
        parent-moral requirements and translates plot-role references from the
        listening child to the established Story World protagonist without
        weakening audience-facing child-readability guidance.
        """
        if not rules:
            return ""

        adapted_lines: list[str] = []
        for raw_line in str(rules).splitlines():
            line = raw_line
            lower = line.lower()

            # Living World has no parent-selected moral. Remove moral-specific
            # instructions rather than leaving contradictory remnants.
            if "moral" in lower:
                continue

            replacements = (
                ("child-led decision", "protagonist-led decision"),
                ("child-led solution", "protagonist-led solution"),
                ("The child must", "The protagonist must"),
                ("the child must", "the protagonist must"),
                ("The child should", "The protagonist should"),
                ("the child should", "the protagonist should"),
                ("The child's", "The protagonist's"),
                ("the child's", "the protagonist's"),
                ("where the child started", "where the protagonist started"),
                ("why the child is involved", "why the protagonist is involved"),
                ("because A happened, the child did B", "because A happened, the protagonist did B"),
                ("with the child", "with the protagonist"),
                ("by the child", "by the protagonist"),
                ("the child discovers", "the protagonist discovers"),
                ("the child notices", "the protagonist notices"),
                ("the child remembers", "the protagonist remembers"),
                ("the child says", "the protagonist says"),
                ("the child does", "the protagonist does"),
                ("the child keeps", "the protagonist keeps"),
                ("the child gives", "the protagonist gives"),
                ("the child or supporting character", "the protagonist or supporting character"),
                ("the child and important characters", "the protagonist and important characters"),
                ("child or helper", "protagonist or helper"),
                ("Return the child", "Return the protagonist"),
                ("return the child", "return the protagonist"),
                ("Give the child an emotional reward", "Give the protagonist an emotional reward"),
                ("give the child an emotional reward", "give the protagonist an emotional reward"),
                ("The child and parent should", "The listener and parent should"),
                ("the child and parent should", "the listener and parent should"),
                ("for the child", "for the listener"),
            )
            for old, new in replacements:
                line = line.replace(old, new)

            adapted_lines.append(line)

        return "\n".join(adapted_lines).strip()

    def _living_world_depth_contract(self, age: Any) -> str:
        """Age-aware page depth derived from the original 115-155 word contract."""
        child_age = self._safe_child_age(age)
        if child_age <= 4:
            target = "55-80"
        elif child_age <= 6:
            target = "75-105"
        elif child_age <= 8:
            target = "105-135"
        elif child_age <= 10:
            target = "115-145"
        else:
            target = "120-155"

        return f"""LIVING WORLD STORY DEPTH — INHERITED FROM THE ORIGINAL PILLOWTALES ENGINE:
- Create exactly 7 substantial pages.
- Each page should normally contain about {target} words.
- Each page should normally contain 5-7 read-aloud sentences in 2 short paragraphs.
- Do not create one strong opening followed by thin summary pages.
- Do not pad with repeated description, recap, wandering, or extra lore merely to reach length.
- Earn the length through action, dialogue, character behaviour, clues, choices, complications, consequences and callbacks.
- Every page must contain a meaningful story beat that changes what happens next.
- One real complication or setback is mandatory before the decisive resolution.
- At least one supporting character should have a distinctive job, habit, phrase, worry, tool or behaviour that affects the plot.
- Include at least one memorable visual, magical or funny moment that a child could describe tomorrow.
- Reuse at least one early story detail later with purpose.
"""

    def _living_world_inherited_quality_rules(self, request: GenerateStoryRequest) -> str:
        """Compile compatible mature PillowTales rules for Story Worlds."""
        story_flow = self._living_world_adapt_original_rules(self._story_flow_rules())
        literary = self._living_world_adapt_original_rules(self._literary_polish_rules())
        clarity = self._living_world_adapt_original_rules(self._story_clarity_rules())
        character_memory = self._living_world_adapt_original_rules(self._character_memory_rules())
        emotional_cohesion = self._living_world_adapt_original_rules(self._emotional_cohesion_rules())
        ending = self._living_world_adapt_original_rules(self._ending_engine_rules())

        return f"""ORIGINAL PILLOWTALES QUALITY ENGINE — ACTIVE FOR THIS LIVING WORLD EPISODE:

{self._oxford_inspired_age_profile_block(request.age)}

{self._age_readability_block(request.age)}

{self._age_vocabulary_block(request.age)}

{self._age_quality_control_block(request.age)}

{clarity}

{story_flow}

{character_memory}

{emotional_cohesion}

{literary}

SHOW, DON'T EXPLAIN — LIVING WORLD:
- Do not narrate conclusions the scene has already shown.
- Avoid summary sentences such as "This was a deeper problem than they expected", "This was different from anything they had seen", or "The situation was becoming serious".
- Show escalation through what changes: a failed attempt, a character reaction, a consequence, a silence, a damaged plan, a new obstacle, or short dialogue.
- Prefer a visible or audible consequence over an abstract explanation.
- Let the listener infer simple emotions and significance at the level appropriate to the Oxford-inspired age profile.

{self._living_world_depth_contract(request.age)}

LIVING WORLD ADAPTATION:
- Wherever the original PillowTales engine expects the listening child to make the choice or solve the problem, the established Story World protagonist performs that role instead.
- The listening child remains completely outside the plot.
- Parent-selected theme and parent-selected moral are disabled.
- Story World Source Canon, continuity, protected names, geography, relationships, institutions, creatures and powers override generic creative suggestions.
- A new detail invented for this episode is story-local unless continuity explicitly marks it as persistent. Do not write a one-off invention as though it has always been established world canon.
- Write the finished story entirely in the requested story language. Preserve protected proper names exactly, but translate ordinary descriptive wording and non-protected invented labels naturally.
- Do not modernise the world merely because the premise resembles a modern situation. Express institutions and conflicts in language and imagery that belong naturally to this Story World.
- Avoid adult labels such as politics, bureaucracy, policy, administration or governance when a child-friendly world-specific description can show the same idea.
- The world may contain leadership contests, councils, promises, rules, disagreements, danger, mysteries, rivals or defence, but present them as story events rather than adult explanation.

ENDING QUALITY — ADAPTED FROM THE ORIGINAL ENGINE:
{ending}
"""

    def _living_world_age_style_block(self, age: Any) -> str:
        """Age calibration specifically for Story World Living World episodes.

        This is prompt-only and does not change page count, Page-1-first,
        background generation, narration, polling, storage, or reader flow.
        """
        child_age = self._safe_child_age(age)

        if child_age <= 4:
            return """LIVING WORLD AGE STYLE — AGE 0-4:
- Use very simple read-aloud language and one obvious problem.
- Most sentences should be 5-9 words.
- Use concrete actions and familiar feelings.
- Avoid lore-heavy explanation, abstract ideas, politics, symbolism, and long descriptions.
- Keep named characters and locations to the minimum needed to follow the episode."""
        if child_age <= 6:
            return """LIVING WORLD AGE STYLE — AGE 5-6:
- Use plain early-reader adventure language.
- Most sentences should be 5-10 words.
- Use one action or idea per sentence.
- Prefer common verbs and concrete nouns.
- Avoid abstract phrases, formal language, layered motives, and lore explanations.
- If a sentence sounds impressive but harder to understand, simplify it."""
        if child_age <= 8:
            return """LIVING WORLD AGE STYLE — AGE 7-8:
- Write for an eight-year-old listening at bedtime, not for a ten-year-old reader.
- Most sentences should usually be 10-18 words, matching the Oxford-inspired age profile. Shorter sentences are welcome for pace and dialogue.
- Prefer everyday words and concrete actions over abstract or literary wording.
- Keep dialogue short and natural.
- Avoid phrases such as "in their everyday concerns", "familiar longing", "urgent sense of purpose", "shared a silent worry", "magical imbalance", "protective measure", or similar adult-sounding abstractions.
- Explain unusual world rules through what characters see and do, not through long narration.
- Use no more than one important new idea per paragraph.
- A child should understand what happened on the first listen without needing a word explained.
- Rich folklore names and places may remain exact; simplify the surrounding English instead."""
        if child_age <= 10:
            return """LIVING WORLD AGE STYLE — AGE 9-10:
- Use clear middle-grade bedtime language with controlled detail.
- Most sentences should usually be 12-22 words, matching the Oxford-inspired age profile, while keeping read-aloud clarity.
- Allow richer motives and world detail, but keep the central action easy to follow.
- Prefer active scenes and dialogue over abstract explanation."""
        return """LIVING WORLD AGE STYLE — AGE 11-12:
- Use richer children's fiction language while remaining read-aloud friendly.
- Allow more nuance, but avoid adult literary density.
- Keep the main problem and cause-and-effect clear on every page."""

    def _build_folk_adventure_first_page_prompt(
        self,
        request: GenerateStoryRequest,
        companion: Optional[dict],
    ) -> str:
        blocks = self._language_and_character_blocks(request, companion)
        child_age = self._safe_child_age(request.age)
        max_chars = 650 if child_age <= 6 else 900 if child_age <= 8 else 1050
        seed = self._select_living_world_episode_seed(request) or {}
        return f"""Write Page 1 only of a PillowTales Living World episode. Return only final JSON.

LANGUAGE:
- Write only in {blocks['language_name']}.
- {self._first_page_language_style_block(request.storyLanguageCode)}

LISTENER PROFILE:
- Listening child: {request.childName}
- Listening age: {request.age}
- The child is outside the adventure itself.
- The child's name may appear ONLY in the external bedtime invitation at the very start of Page 1.
- After that invitation, never make the listener a character, witness, dreamer, helper, narrator, visitor or participant.
- No moral is requested.

{self._story_world_prompt_block(request)}

{self._living_world_inherited_quality_rules(request)}

{self._natural_name_pronoun_rules(protect_canon_names=True)}

PAGE 1 LIVING WORLD JOB:
- Use this selected episode seed as the central premise: {json.dumps(seed, ensure_ascii=False)}
- FIRST PARAGRAPH ONLY: write a brief 1-2 sentence personalised bedtime invitation using {request.childName}. Invite the child to snuggle down/get cosy and get ready to hear or visit this Story World. Write the invitation naturally in {blocks['language_name']}; do not translate protected Story World names.
- SECOND PARAGRAPH: begin the actual episode immediately inside the selected Story World with an established or continuity-approved world character taking action.
- Choose the protagonist only from Source Canon or Living World continuity. Never hardcode a character from another Story World.
- After the first paragraph, do not mention or refer to the listening child anywhere in the episode.
- The bedtime invitation is external framing only; the actual plot must not start in the child's bedroom, ordinary beach, home or dream.
- Do not use whispering waves, humming shells, shimmering objects, forgotten memories, lost wishes, magical fragments, ribbons or vague calls for help.
- Establish a concrete world-specific disruption, opportunity, discovery, rivalry, event or threat.
- Make the selected Story World operationally necessary to what happens.
- The anchor characters, powers, places, institutions or continuity must materially affect the plot.
- Give the protagonist a clear immediate objective by the end of Page 1.
- Do not solve it yet.
- Title the episode around the world character, place, event or problem; do not automatically put the listening child's name in the title.

AGE AND LENGTH:
{self._living_world_age_style_block(request.age)}
- Maximum {max_chars} characters for this fast Page 1 response.
- Aim for a substantial opening rather than a synopsis: normally 5-7 read-aloud sentences in 2 short paragraphs.
- Establish character, place, concrete problem and immediate objective through scene action rather than explanation.
- Adventure can be exciting; bedtime-safe does not mean passive or overly gentle.

JSON ONLY:
{{"title":"...","pages":["page 1 text"]}}
"""

    def _build_first_page_prompt(self, request: GenerateStoryRequest, companion: Optional[dict]) -> str:
        if self._is_canon_request(request):
            return self._build_canon_first_page_prompt(request, companion)
        if self._is_folk_adventure_request(request):
            return self._build_folk_adventure_first_page_prompt(request, companion)

        blocks = self._language_and_character_blocks(request, companion)

        opening_seed = self._select_opening_seed(request)
        opening = opening_seed["sentence"]
        language_code = (request.storyLanguageCode or "en").lower()[:2]
        child_age = self._safe_child_age(request.age)
        story_world_block = self._story_world_prompt_block(request)

        if child_age <= 2:
            age_contract = "Use baby/toddler read-aloud language: very short sentences, familiar words, one place, one tiny event."
            max_chars = "420"
        elif child_age <= 5:
            age_contract = "Use preschool bedtime language: short clear sentences, one place, one small funny or curious problem, obvious cause and effect."
            max_chars = "520"
        elif child_age <= 6:
            age_contract = "Use age-6 early-reader language: everyday words, 5-10 word sentences, one clear problem or wish, no poetic or abstract wording."
            max_chars = "560"
        elif child_age <= 8:
            age_contract = "Use young-child adventure language: clear sentences, one memorable character idea, one goal, no crowded setup."
            max_chars = "620"
        else:
            age_contract = "Use older-child bedtime language: richer but clear, one hook, one reason the child is involved."
            max_chars = "700"

        return f"""Write Page 1 only. Do not plan or analyse. Return only final JSON.

LANGUAGE:
- Write only in {blocks['language_name']}.
- {self._first_page_language_style_block(language_code)}

STORY FACTS:
- Child: {request.childName}
- Internal reading age: {request.age}. This controls writing level only. Never state the numerical age in the story prose.
- Theme: {blocks['effective_theme']}
- Moral: {request.moral}

STORY WORLD ISOLATION:
- This is a standard PillowTales story unless Story World context is explicitly present below.
- Do not import characters, places, canon, continuity, terminology or institutions from any Story World into a standard story.

{story_world_block}

{self._children_author_voice_rules()}
{self._first_page_spine_setup_rules(request)}
{self._natural_name_pronoun_rules()}
{self._standard_bedtime_first_page_quality_rules(request)}
PAGE 1 JOB:
- Write like a relaxed children's author starting a favourite bedtime adventure.
- Start from this theme-matched opening idea, rewritten naturally: "{opening}"
- Show who the story is about, where they are, and what makes today different.
- Give the story one clear child-friendly hook: a funny wish, silly mistake, odd visitor, simple problem, useful mystery, or goal the child wants to achieve.
- The hook should fit the chosen theme and make the child want to hear Page 2.
- Treat the selected theme as a promise to the child. Introduce its central subject or a direct path to it now; do not save the main themed character or experience for the final page.
- Give the hook real story value; do not rely only on a glowing object or unexplained magical light.
- Include one small memorable detail a child might repeat tomorrow.
- Keep one clear next step for Page 2.
- Do not solve the story yet.

AGE CONTRACT:
- {age_contract}

STYLE LIMITS:
- Maximum {max_chars} characters for page text.
- 4-6 read-aloud sentences.
- Use simple words and concrete actions.
- Make it sound like a parent reading a picture book, not an adult fantasy novel.
- Harmless silliness is welcome if it fits the story.
- Avoid decorative overload and repeated AI words such as glowing, shimmering, sparkling, moonlit, softly, slowly, sleepy.
- Do not describe the child as little, small, tiny, young, or physically childlike.
- Do not write phrases like small hands, little feet, little girl, little boy, or young explorer.
- Refer to the child by name or pronouns.

COMPANION:
- {blocks['companion_line']}

CHARACTERS:
- {blocks['character_instruction']}

JSON ONLY:
- Story text must be plain prose only: no Markdown, asterisks, underscores, headings, bullets, backticks, or formatting notation.
{{"title":"The Sleepy Dragon","pages":["page 1 text"]}}
- Replace "The Sleepy Dragon" with a real short title for this story.
- Never return placeholder titles such as "Short title", "Title", or "Story Title".
"""

    def _build_living_world_first_page_fallback(self, request: GenerateStoryRequest, companion: Optional[dict]) -> Dict[str, Any]:
        """Instant local fallback that stays inside the selected Living World.

        This fallback is generic across Story Worlds and must never hardcode
        Ireland, Tír na nÓg, Niamh, Oisín or any first-world identity.
        """
        context = self._resolve_story_world_context(request) or {}
        continuity = (context.get('living_world_continuity') or {}).get('content') or {}
        seed = self._select_living_world_episode_seed(request, context) or {}
        language_code = (request.storyLanguageCode or "en").lower()[:2]
        child = str(request.childName or "the child").strip() or "the child"

        characters = continuity.get('persistent_characters') or {}
        protagonist = ""
        if isinstance(characters, dict) and characters:
            protagonist = str(next(iter(characters.keys())))
        elif isinstance(characters, list) and characters:
            first = characters[0]
            protagonist = str(first.get('name') if isinstance(first, dict) else first)
        if not protagonist:
            anchor_characters = (context.get('anchor') or {}).get('main_characters') or []
            if isinstance(anchor_characters, list) and anchor_characters:
                first = anchor_characters[0]
                protagonist = str(first.get('name') if isinstance(first, dict) else first)
        protagonist = protagonist.strip() or {
            "es": "una guardiana del lugar",
            "fr": "une gardienne du lieu",
            "de": "eine Hüterin dieser Welt",
            "it": "una custode di quel mondo",
        }.get(language_code, "a guardian of the world")

        invitations = {
            "en": f"Snuggle down, {child}, and get nice and cosy. Tonight, a new Story World adventure is ready for you.",
            "es": f"Acurrúcate bien, {child}, y ponte cómodo. Esta noche te espera una nueva aventura en Story Worlds.",
            "fr": f"Installe-toi bien, {child}, et mets-toi à l'aise. Ce soir, une nouvelle aventure de Story Worlds t'attend.",
            "de": f"Kuschel dich ein, {child}, und mach es dir gemütlich. Heute Abend wartet ein neues Story-World-Abenteuer auf dich.",
            "it": f"Rannicchiati bene, {child}, e mettiti comodo. Stasera ti aspetta una nuova avventura di Story Worlds.",
        }
        generic_openings = {
            "en": f"Inside that world, {protagonist} noticed that something familiar had changed and set off to find out why.",
            "es": f"Dentro de aquel mundo, {protagonist} vio que algo conocido había cambiado y salió a descubrir por qué.",
            "fr": f"Dans ce monde, {protagonist} remarqua que quelque chose de familier avait changé et partit découvrir pourquoi.",
            "de": f"In dieser Welt bemerkte {protagonist}, dass sich etwas Vertrautes verändert hatte, und machte sich auf den Weg, den Grund herauszufinden.",
            "it": f"In quel mondo, {protagonist} si accorse che qualcosa di familiare era cambiato e partì per capire il perché.",
        }

        title = str(seed.get('title') or '').strip()
        if language_code != "en" or not title:
            title = {
                "es": "Una aventura en Story Worlds",
                "fr": "Une aventure dans Story Worlds",
                "de": "Ein Abenteuer in Story Worlds",
                "it": "Un'avventura in Story Worlds",
            }.get(language_code, "A Story World Adventure")

        page = (
            f"{invitations.get(language_code, invitations['en'])}\n\n"
            f"{generic_openings.get(language_code, generic_openings['en'])}"
        )
        return {
            'title': title,
            'pages': postprocess_story_pages([page])[:1],
            'companion': companion,
            'expected_pages': self._intended_page_count(request),
            'generation_status': 'partial',
            'generation_fallback_reason': 'living_world_first_page_fallback',
            'first_page_generation_source': 'fallback_living_world_local',
        }

    def _build_living_world_remaining_pages_prompt(
        self,
        request: GenerateStoryRequest,
        companion: Optional[dict],
        title: str,
        existing_pages: list[str],
        remaining_page_count: int,
        next_page_number: int,
    ) -> str:
        blocks = self._language_and_character_blocks(request, companion)
        existing_pages_text = "\n\n".join(f"Page {idx + 1}: {page}" for idx, page in enumerate(existing_pages or []))
        final_page_number = next_page_number + remaining_page_count - 1
        is_final = final_page_number >= self._intended_page_count(request)
        role_map = {
            2: (
                "DEEPEN THE PROMISE. Develop the exact Page 1 problem or opportunity. "
                "Introduce only one useful clue, helper, obstacle or world rule. "
                "Give the protagonist a clear next step."
            ),
            3: (
                "FIRST REAL COMPLICATION. Let the first approach partly fail, expose a new difficulty, "
                "or reveal that the problem is not as simple as it looked. "
                "The protagonist must notice, ask, choose or try something that changes what happens next."
            ),
            4: (
                "MIDPOINT TURN. Reveal the strongest clue, hidden motive, surprising truth or world rule so far. "
                "Change the protagonist's understanding or plan. Do not solve the whole problem here."
            ),
            5: (
                "STRONGEST SETBACK. The protagonist acts on what was learned, but meets the hardest obstacle, "
                "failed attempt, difficult choice or reversal. Success should briefly feel uncertain. "
                "Bring back an earlier clue, habit, promise, object, joke or world rule and make it matter."
            ),
            6: (
                "DECISIVE ACTION AND CLIMAX. The protagonist drives the solution using established clues, "
                "relationships, skills, powers, strategy or teamwork. This must be the adventure's peak. "
                "The main problem should be resolved or be visibly and irreversibly resolving by the end."
            ),
            7: (
                "FINAL PAYOFF AND BEDTIME LANDING. If one final action or reveal is still required to complete "
                "the climax, finish it immediately using only established story material. Then show consequence, "
                "relief, one earned callback and a calm settled ending. No new problem, quest, clue, magical rule or sequel hook."
            ),
        }
        ending = (
            "This is the final page. Complete any already-established final resolution immediately, then give the result room to breathe. Do not invent a new solution, cause, character, place, magical rule, task or sequel hook. End exactly with The End."
            if is_final else
            "Do not finish the whole episode prematurely; follow the page role and keep the same central problem."
        )
        return f"""Continue this PillowTales Living World episode. Return only JSON.

LANGUAGE:
- Write only in {blocks['language_name']}.

{self._story_world_prompt_block(request)}

{self._living_world_inherited_quality_rules(request)}

{self._natural_name_pronoun_rules(protect_canon_names=True)}

FIXED EPISODE SPINE:
{self._story_spine_block(request, title, (existing_pages or [''])[0])}

EXISTING PAGES:
{existing_pages_text}

PAGE {next_page_number} ROLE:
- {role_map.get(next_page_number, 'Move the same episode forward.')}

CONTINUATION RULES:
- Write exactly {remaining_page_count} new page(s), Page {next_page_number} through Page {final_page_number}.
- Preserve the protagonist established after the external bedtime invitation on Page 1.
- Treat Page 1's first-paragraph bedtime invitation as external framing, not as part of the episode plot.
- The listening child must NEVER appear, be named, be addressed, or be inserted at any later point.
- Write all ordinary narration, dialogue and non-protected descriptive labels naturally in {blocks['language_name']}. Preserve only protected proper names exactly.
- Continue the selected episode seed and the same concrete world-specific problem.
- Every page must materially use established locations, characters, powers, institutions, creatures or rules from continuity.
- Do not drift into a generic object hunt, magical repair, lost memory, lost wish, shell, ribbon, whisper, shimmer or moral lesson.
- Do not make gentleness or kindness the automatic solution.
- Magic cannot automatically solve the plot; action, intelligence, courage, strategy or teamwork must matter.
{self._living_world_age_style_block(request.age)}
- Keep each sentence focused on one clear action, observation, or spoken idea.
- Follow the Living World story-depth target above. Do not return a synopsis-sized page.
- Use action, dialogue, complication, character behaviour and consequence to create depth; never pad with recap or decorative lore.
- Each page should normally contain 5-7 read-aloud sentences in 2 short paragraphs.
- A command from the protagonist must not end the conflict by itself. If an antagonist, rival or obstacle matters, require a believable action, choice, trick, cost, discovery, reversal or consequence before resolution.
- {ending}

JSON ONLY:
{{"pages":["new page text"]}}
- Return exactly {remaining_page_count} string(s), with no notes or extra keys.
"""

    def _build_first_page_fallback(self, request: GenerateStoryRequest, companion: Optional[dict]) -> Dict[str, Any]:
        if self._is_canon_request(request):
            return self._build_canon_first_page_fallback(request, companion)
        if self._is_folk_adventure_request(request):
            return self._build_living_world_first_page_fallback(request, companion)

        """Fast polished page-1 fallback used only when Gemini is too slow or malformed.

        This must remain instant and local. It protects the Page-1-first
        architecture without making fallback feel like a repeated placeholder.
        Do not call Gemini here and do not generate the full story here.
        """
        child = request.childName or "the child"
        language_code = (request.storyLanguageCode or "en").lower()[:2]
        expected_pages = self._intended_page_count(request)
        theme = self._localized_theme_label(request.theme, language_code) or "magic"
        theme_key = str(request.theme or "").lower().replace("-", "_").replace(" ", "_")
        localized_companion = self._localized_companion(companion, language_code)

        opening_seed = self._select_opening_seed(request)
        opening_sentence = opening_seed.get("sentence") or f"{child} found something unusual just before story time."

        companion_en = ""
        if localized_companion:
            companion_en = f" {localized_companion['name']} came too, keeping the first clue safe."

        # Dynamic English fallback: many combinations, all local/instant.
        # This avoids repeated starts such as Star Signal / Pawprint Parade when
        # Gemini times out, while preserving the same Page-1-first contract.
        english_problem_pool = {
            "dragons": [
                "A soot-smudged postcard said the dragon post office had mixed up every bedtime letter.",
                "A teacup-sized dragon had practised a roar so small that nobody at the mountain market had heard it.",
                "A trail of warm smoke curls pointed toward a dragon hatchery where one egg had started humming at the wrong time.",
            ],
            "space": [
                "The telescope tapped three times, and a message on the lens said the sky train had too many wishes and not enough seats.",
                "A paper comet slid under the door with a note asking for help sorting tomorrow's constellations.",
                "A star map folded itself into a bird and dropped one blinking dot into {child}'s hand.",
            ],
            "animals": [
                "A line of pawprints crossed the floor toward a rabbit who had brought the parade drum but forgotten the first beat.",
                "A hedgehog messenger arrived with a badge on backwards and a list of animals who all wanted to lead at once.",
                "A squirrel in an oversized hat whispered that the garden band had lost the quietest instrument.",
            ],
            "princess": [
                "A paper crown slid from a cushion tower with one blank side waiting for a fair idea.",
                "An invitation from the royal garden said two friends both wanted to carry the first lantern.",
                "A ribbon from the castle parade wriggled free and tied itself into a question mark.",
            ],
            "adventure": [
                "A folded map showed the room, the doorway, and one path that definitely had not been there before.",
                "A brass button rolled from under the bedtime book and stopped beside a drawn arrow.",
                "A small sign appeared on the floorboards saying, 'One helpful traveller needed before sunset.'",
            ],
        }
        generic_problem_pool = [
            f"A folded message marked with a picture from a {theme} place said two friends needed help before the evening settled.",
            f"A crooked thread tied around a small note pointed toward a {theme} problem waiting just beyond the room.",
            f"A tiny sign showed two hands holding the same ribbon, as if the adventure could only begin when someone chose to share.",
            f"A small visitor had left a question beside the bedtime book, and the question seemed to belong to {child} now.",
            f"A quiet sound came from the doorway, followed by a clue that looked too deliberate to ignore.",
        ]
        english_problem = random.choice(english_problem_pool.get(theme_key, generic_problem_pool)).format(child=child)
        english_actions = [
            f"{child} looked at the clue, asked one careful question, and chose to follow it before the trail disappeared.",
            f"{child} kept the first clue close, promised to listen before rushing, and took the first step.",
            f"{child} noticed the odd detail nobody else had seen and gently opened the way forward.",
            f"{child} touched the clue, remembered to be brave in a small way, and followed where it pointed.",
        ]
        title_bits = {
            "dragons": ["Dragon Bell", "Teacup Dragon", "Small Roar", "Post Office Dragon"],
            "space": ["Star Map", "Sky Train", "Paper Comet", "Telescope Message"],
            "animals": ["Pawprint Parade", "Garden Band", "Rabbit's Ribbon", "Hedgehog Message"],
            "princess": ["Shared Crown", "Ribbon Parade", "Cushion Castle", "Lantern Walk"],
            "adventure": ["Map Under the Book", "Crooked Thread", "First Clue", "Helpful Path"],
        }
        english_titles = title_bits.get(theme_key, [f"{theme.title()} Promise", "First Clue", "Crooked Thread", "Bedtime Message"])

        english_variant = {
            "title": f"{child} and the {random.choice(english_titles)}",
            "page": f"{opening_sentence} {english_problem} {random.choice(english_actions)}{companion_en}",
        }

        # Non-English fallbacks deliberately use the selected local opening seed
        # so they also vary without adding network calls or touching narration.
        fallback_variants = {
            "en": [english_variant],
            "es": [
                {
                    "title": f"La primera pista de {child}",
                    "page": (
                        f"{opening_sentence} "
                        f"Junto al cuento de dormir apareció un mensaje doblado, con un dibujo de {theme} y un hilo torcido en una esquina. "
                        f"Decía que dos amigos necesitaban ayuda antes de que terminara la tarde. "
                        f"{child} guardó el hilo como una promesa y siguió la primera pista hacia la puerta."
                    ),
                }
            ],
            "fr": [
                {
                    "title": f"Le premier indice de {child}",
                    "page": (
                        f"{opening_sentence} "
                        f"Près du livre du soir attendait un message plié, avec un dessin de {theme} et un fil de travers dans un coin. "
                        f"Il disait que deux amis avaient besoin d'aide avant la fin du soir. "
                        f"{child} garda le fil comme une promesse et suivit le premier indice vers la porte."
                    ),
                }
            ],
            "de": [
                {
                    "title": f"{child}s erster Hinweis",
                    "page": (
                        f"{opening_sentence} "
                        f"Neben dem Gute-Nacht-Buch lag eine gefaltete Nachricht mit einem Bild von {theme} und einem schiefen Faden an der Ecke. "
                        f"Darin stand, dass zwei Freunde vor dem Abend Hilfe brauchten. "
                        f"{child} bewahrte den Faden wie ein Versprechen auf und folgte dem ersten Hinweis zur Tür."
                    ),
                }
            ],
            "it": [
                {
                    "title": f"Il primo indizio di {child}",
                    "page": (
                        f"{opening_sentence} "
                        f"Accanto al libro della sera c'era un messaggio piegato, con un disegno di {theme} e un filo storto legato a un angolo. "
                        f"Diceva che due amici avevano bisogno di aiuto prima che finisse la sera. "
                        f"{child} tenne il filo come una promessa e seguì il primo indizio verso la porta."
                    ),
                }
            ],
            "ja": [
                {
                    "title": f"{child}とはじめの手がかり",
                    "page": (
                        f"{opening_sentence} "
                        f"ねる前の絵本のそばに、{theme}の絵がかかれた小さな手紙が置いてありました。 "
                        f"手紙には、夜になる前に二人の友だちを助けてほしいと書いてありました。 "
                        f"{child}は手紙を大切に持ち、最初の手がかりをたどってドアのほうへ進みました。"
                    ),
                }
            ],
            "ar": [
                {
                    "title": f"{child} والدليل الأول",
                    "page": (
                        f"{opening_sentence} "
                        f"وبجوار كتاب ما قبل النوم، ظهرت رسالة صغيرة عليها صورة عن {theme}. "
                        f"قالت الرسالة إن صديقين يحتاجان إلى المساعدة قبل أن يحل المساء. "
                        f"احتفظ {child} بالرسالة بعناية، ثم اتبع الدليل الأول نحو الباب."
                    ),
                }
            ],
        }

        variants = fallback_variants.get(language_code, fallback_variants["en"])
        selected = random.choice(variants)
        pages = postprocess_story_pages([selected["page"]])[:1]
        print(
            f"[PERF] first_page_fallback_selected title={selected['title']!r} "
            f"lang={language_code} theme={theme_key or theme!r} opening_family={opening_seed.get('family')}"
        )
        return {
            'title': selected['title'],
            'pages': pages,
            'companion': companion,
            'expected_pages': expected_pages,
            'generation_status': 'partial',
            'generation_fallback_reason': 'first_page_fallback',
            'first_page_generation_source': 'fallback_local_dynamic',
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
        if self._is_canon_request(request):
            return self._build_canon_remaining_pages_prompt(
                request=request,
                companion=companion,
                title=title,
                existing_pages=existing_pages,
                remaining_page_count=remaining_page_count,
                next_page_number=next_page_number,
            )
        if self._is_folk_adventure_request(request):
            return self._build_living_world_remaining_pages_prompt(
                request=request,
                companion=companion,
                title=title,
                existing_pages=existing_pages,
                remaining_page_count=remaining_page_count,
                next_page_number=next_page_number,
            )

        """Build a compact continuation prompt for pages 2+.

        The continuation prompt is intentionally simpler than earlier Phase 11
        versions. It keeps the working story structure but forces child-level
        language, character-first fun, and real page length.
        """
        blocks = self._language_and_character_blocks(request, companion)
        age_rules = self._first_page_age_prompt_rules(request.age)
        existing_pages_text = "\n\n".join(
            f"Page {idx + 1}: {page}" for idx, page in enumerate(existing_pages or [])
        )
        story_world_block = self._story_world_prompt_block(request)
        story_spine = self._story_spine_block(
            request=request,
            title=title,
            first_page=(existing_pages or [""])[0],
        )
        final_page_number = next_page_number + remaining_page_count - 1
        intended_final_page = self._intended_page_count(request)
        page_role = self._page_narrative_role(next_page_number)
        page_tension = self._page_tension_rules(next_page_number)
        includes_final_page = next_page_number <= intended_final_page <= final_page_number
        if includes_final_page:
            ending_job = f"""FINAL PAGE MODE — THIS STORY ENDS NOW:
- You are writing ONLY the ending of the story shown above.
- Do not continue the adventure as though another page follows.
- Everything must stay in tune with this story's existing goal, characters, setting, humour, promises, and magical rules.
- Resolve what Pages 1-6 actually set up. Do not replace it with a generic celebration or bedtime paragraph.
- The final page should move through: resolution, brief story-specific afterglow, safe settling, final image.
- Do not add a new character, named friend, place, task, object, clue, problem, secret, or surprise.
- Do not summarise the ending with "felt brave", "felt proud", "felt like the bravest", or similar wording.
- The final story sentence must be a concrete image, action, sound, or callback from Pages 1-6.
- The page must feel complete even before the words The End.
- Finish with exactly: The End.

{self._literary_polish_rules()}
{self._ending_engine_rules()}"""
        elif next_page_number == intended_final_page - 1:
            ending_job = """PENULTIMATE PAGE — PAGE 6:
- This is the final challenge and only one page will remain.
- Bring the main problem to its decisive moment and begin resolving it here.
- Do not introduce a new character, location, subplot, mystery, magical rule, or second problem.
- Do not stretch the same activity across Page 6 and Page 7.
- Leave Page 7 to show the completed result, emotional payoff, callback, and safe closing.
"""
        else:
            ending_job = """NOT THE FINAL PAGE:
- Do not resolve the whole story yet.
- Progress the main problem and leave one clear, bedtime-safe reason to continue.
- Do not introduce a new subplot, unrelated mystery, or premature return-home ending.
"""

        story_identity_line = (
            "Story identity: selected Story World + selected folklore source. Ignore the parent's generic theme."
            if self._is_folk_adventure_request(request)
            else f"Theme: {blocks['effective_theme']}"
        )

        folk_adventure_continuity = (
            """FOLK ADVENTURE CONTINUITY — NON-NEGOTIABLE:
- Keep the same selected folklore source dependency established on Page 1.
- Every major development must remain connected to that source's place, consequence, untold space, protected fact, or compatible character role.
- Do not drift into a generic fantasy mission that could be moved to another country by renaming places.
- Prefer the supplied Story World landscape, customs, creatures, objects, atmosphere, places, relationships and folklore consequences over generic fantasy inventions.
- Do not invent a missing, forgotten, secret, corrected, repaired, recovered, or alternative part of the source legend.
- Do not rewrite, prevent, reverse, repair, complete, restore, or replace any protected canonical event or outcome.
- The child solves only the separate NEW mission and is never responsible for making canon happen correctly.
- If a generic creative rule conflicts with Story World authenticity or the source contract, the Story World material and source contract win.
- Preserve protected cultural names exactly.
"""
            if self._is_folk_adventure_request(request)
            else ""
        )

        return f"""Continue this bedtime story from the existing pages.

LANGUAGE:
- Write ONLY in {blocks['language_name']}.
- Do not mix languages.
- Use natural read-aloud bedtime language.
- Sound like a relaxed, funny, warm children's storyteller who knows how to hold a child's attention, not an AI.

STORY FACTS:
- Title: {title}
- Child: {request.childName}
- Internal reading age: {request.age}. Use it only for writing calibration; never state the numerical age in story prose.
- {story_identity_line}
- Moral: {request.moral}

{story_world_block}

{folk_adventure_continuity}

{story_spine}
{self._children_author_voice_rules() if not self._is_folk_adventure_request(request) else ''}
{self._standard_bedtime_elite_quality_rules(request) if not self._is_folk_adventure_request(request) else ''}
EXISTING PAGES:
{existing_pages_text}

AGE LOCK:
{age_rules}
- Keep every page suitable for the internal reading age {request.age}, but never mention that number in the story itself.
- For age 6 and under, use short plain sentences and everyday words.
- Do not use adult, poetic, cinematic, symbolic, or fantasy-novel language.
- If a sentence sounds impressive, simplify it.

STORY FLOW:
{self._story_flow_rules()}

{self._natural_name_pronoun_rules()}

LITERARY POLISH:
{self._literary_polish_rules()}

CURRENT PAGE ROLE — NON-NEGOTIABLE:
{page_role}

CURRENT PAGE TENSION — NON-NEGOTIABLE:
{page_tension}

CONTINUATION JOB:
- Write exactly {remaining_page_count} new pages: Page {next_page_number} through Page {final_page_number}.
- Continue from the latest existing page. Do not recap or contradict it.
- Keep one main story idea visible.
- Each page must clearly move the story forward. It may arrive somewhere, reveal something, test an idea, force a choice, create a setback, solve part of the problem, or settle after the result. These are examples, not a fixed sequence.
- Every page should include something a child can picture or remember.
- Pages before the resolution should end with a small reason to keep listening: a clue, choice, surprise, funny complication, or clear next step.
- Do not use frightening danger or an unresolved cliffhanger.
- Do not let several pages only search, wait, look around, or explain. Something must change on every page.
- Compare the last two existing pages before writing. If they use the same type of scene, this page MUST break that pattern rather than repeat it a third time.
- Use short dialogue, action, and funny behaviour rather than narrator explanation.
- Harmless silliness is allowed if it helps the story.
- Include at least one memorable magical detail that affects the story.
- Include at least one warm funny or surprising moment that changes what happens next.
- The child must help drive the solution.
- Helpers may guide, misunderstand, or make funny mistakes, but they must not solve everything.
- Bring back one earlier detail when it becomes useful or emotionally meaningful.
- Show the moral through action. Do not repeat it after each scene, lecture, or write "learned that".

ENDING JOB FOR THIS BATCH:
{ending_job}

STRICT LANGUAGE CLEANUP:
- Avoid overusing: tiny, little, small, soft, gentle, golden, silver, shimmering, glowing, sparkling, moonlit, sleepy, softly, slowly.
- Do not describe the child as little, small, tiny, young, or physically childlike.
- Do not write phrases like small hands, little feet, little girl, little boy, or young explorer.
- Refer to the child by name or pronouns.
- Avoid advanced words unless they are truly age-suitable.
- Avoid repeated sentence patterns such as the child looked, the child walked, the child found on every page.

PAGE LENGTH:
- Each continuation page must be a real story page, not a caption or summary.
- Each page should be 105-155 words.
- Minimum acceptable continuation page length is 80 words.
- Never return a single-sentence page.
- Each page should have exactly 2 short paragraphs.
- Each page should contain about 5-7 read-aloud sentences.
- Final page may be slightly shorter if complete, satisfying, and naturally settled.

COMPANION:
- {blocks['companion_line']}

CHARACTERS:
- {blocks['character_instruction']}

OUTPUT FORMAT STRICT:
Return ONLY valid JSON:
{{{{"pages":["new page text","new page text"]}}}}
- The pages array must contain exactly {remaining_page_count} strings.
- No markdown, notes, explanations, or extra keys. Story strings must be plain prose with no asterisks, underscores, headings, bullets, backticks, or formatting notation.
- Silently check that every page progresses the story and feels fun, warm, and age-appropriate.
"""

    def _log_gemini_response_metadata(self, label: str, response: Any) -> None:
        """Best-effort diagnostics for Gemini responses.

        This is logging only. It does not change story generation, parsing,
        narration, polling, subscriptions, Parent Voice, or reader behaviour.
        """
        try:
            candidates = getattr(response, "candidates", None) or []
            finish_reasons: list[str] = []
            for candidate in candidates:
                reason = getattr(candidate, "finish_reason", None)
                if reason is not None:
                    finish_reasons.append(str(reason))

            usage = getattr(response, "usage_metadata", None)
            usage_parts: list[str] = []
            if usage is not None:
                for attr in (
                    "prompt_token_count",
                    "candidates_token_count",
                    "total_token_count",
                    "thoughts_token_count",
                ):
                    value = getattr(usage, attr, None)
                    if value is not None:
                        usage_parts.append(f"{attr}={value}")

            text = getattr(response, "text", None)
            text_len = len(text) if isinstance(text, str) else 0
            print(
                f"[PERF] {label}_metadata candidates={len(candidates)} "
                f"finish_reasons={finish_reasons or 'unknown'} "
                f"text_chars={text_len} usage={' '.join(usage_parts) if usage_parts else 'unknown'}"
            )
        except Exception as log_exc:
            print(f"[PERF] {label}_metadata_log_failed error={str(log_exc)[:200]}")

    def _log_gemini_text_preview(self, label: str, response_text: Any) -> None:
        """Log a short escaped preview when parsing fails.

        The preview is capped to avoid dumping full story content or excessive
        logs. It is only intended to diagnose truncated JSON.
        """
        if not isinstance(response_text, str):
            print(f"[PERF] {label}_raw_preview unavailable type={type(response_text).__name__}")
            return
        preview = response_text[:300].replace("\n", "\\n").replace("\r", "\\r")
        print(f"[PERF] {label}_raw_preview chars={len(response_text)} preview={preview!r}")

    @staticmethod
    def _flatten_canon_text(value: Any) -> list[str]:
        """Return readable strings from nested Canon record values."""
        if value in (None, "", [], {}):
            return []
        if isinstance(value, str):
            return [value.strip()] if value.strip() else []
        if isinstance(value, dict):
            preferred = []
            for key in ("name", "title", "label", "character", "location", "event", "scene", "description", "summary"):
                if key in value:
                    preferred.extend(StoryService._flatten_canon_text(value.get(key)))
            if preferred:
                return preferred
            flattened: list[str] = []
            for item in value.values():
                flattened.extend(StoryService._flatten_canon_text(item))
            return flattened
        if isinstance(value, (list, tuple, set)):
            flattened: list[str] = []
            for item in value:
                flattened.extend(StoryService._flatten_canon_text(item))
            return flattened
        return [str(value).strip()]

    def _canon_page_one_terms(self, request: GenerateStoryRequest) -> list[str]:
        """Extract language-matched Canon anchors for the lightweight Page 1 guard.

        The authoritative Canon contract remains the source of truth. For a
        translated story, the repository already attaches the published story
        translation to the selected Canon anchor; those translated catalogue
        fields are added as matching anchors so Arabic/Japanese prose is not
        rejected merely because its script differs from the base Canon record.
        """
        contract = self._canon_contract(request)
        source_values = [
            contract.get("characters"),
            contract.get("locations"),
            (contract.get("required_scenes") or [])[:1] if isinstance(contract.get("required_scenes"), list) else contract.get("required_scenes"),
            (contract.get("required_events") or [])[:1] if isinstance(contract.get("required_events"), list) else contract.get("required_events"),
        ]

        language = str(request.storyLanguageCode or "en").strip().lower().replace("_", "-").split("-", 1)[0]
        context = self._resolve_story_world_context(request)
        anchor = (context or {}).get("anchor") or {}
        translation = anchor.get("_story_translation") if isinstance(anchor, dict) else None
        if language != "en" and isinstance(translation, dict):
            source_values.extend([
                translation.get("title"),
                translation.get("subtitle"),
                translation.get("summary"),
            ])

        stopwords = {
            "about", "after", "again", "before", "being", "between", "child", "during",
            "first", "from", "into", "legend", "place", "scene", "story", "their", "there",
            "these", "they", "this", "through", "where", "which", "while", "with",
        }
        arabic_stopwords = {
            "التي", "الذي", "هذه", "هذا", "ذلك", "هناك", "كانت", "كان", "بعد", "قبل",
            "عندما", "حيث", "حول", "قصة", "حكاية", "إلى", "على", "من", "في", "مع",
        }
        terms: list[str] = []

        def add_term(term: str) -> None:
            candidate = str(term or "").strip()
            if candidate and candidate not in terms:
                terms.append(candidate)

        for text in self._flatten_canon_text(source_values):
            cleaned = re.sub(r"\s+", " ", text).strip(" -:;,.،؛。！？؟\n\t")
            if 3 <= len(cleaned) <= 80:
                add_term(cleaned)

            for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž’'-]{4,}", cleaned):
                normalized = token.strip("’'-").lower()
                if normalized and normalized not in stopwords:
                    add_term(token)

            if language == "ar":
                for token in re.findall(r"[\u0600-\u06FF]{3,}", cleaned):
                    normalized = token.strip("ـ").lower()
                    if normalized and normalized not in arabic_stopwords:
                        add_term(token)
            elif language == "ja":
                # Prefer compact script runs that are likely to carry names or
                # distinctive title/summary concepts; avoid treating an entire
                # unspaced Japanese sentence as one required phrase.
                for token in re.findall(r"[\u30a0-\u30ff]{2,12}|[\u3400-\u9fff々]{2,8}|[\u3040-\u309f]{3,10}", cleaned):
                    add_term(token)

        return terms[:60]

    @staticmethod
    def _canon_instruction_leak_reason(text: Any) -> Optional[str]:
        """Return a rejection reason if child-facing Canon prose exposes control text.

        This is deliberately narrow and Canon-only. It does not rewrite prose,
        add model calls, wait for later pages, or touch narration. It simply
        prevents editorial/prompt language from being published as story text.
        """
        candidate = re.sub(r"\s+", " ", str(text or "")).strip().lower()
        if not candidate:
            return None

        leaked_phrases = (
            "canonical characters",
            "canonical events",
            "canonical setting",
            "canon characters",
            "canon events",
            "canon setting",
            "canon source of truth",
            "source of truth",
            "canon authority rules",
            "canon layering rule",
            "required canonical event",
            "required canonical events",
            "required event order",
            "generation rules",
            "generation_rule",
            "editorial and cultural boundaries",
            "story world dna",
            "prompt pack",
            "return only json",
            "json only",
            "do not invent",
            "the story continues only with its canonical",
        )
        for phrase in leaked_phrases:
            if phrase in candidate:
                return f"canon_instruction_leak:{phrase.replace(' ', '_')}"
        return None

    def _validate_canon_first_page(self, request: GenerateStoryRequest, story_data: dict) -> tuple[bool, str]:
        """Reject a thin or source-disconnected Canon Page 1 locally.

        For Japanese/Arabic, never require an English Canon substring when the
        catalogue has no published source-scene anchors in the requested script.
        That cross-script requirement caused valid Japanese Canon openings to
        fail closed even though Gemini had generated the correct Tara/Aillen
        scene. In that no-native-anchor case, the exact canonical title plus
        the existing shape guards remain the local identity check; the full
        Canon contract still governs generation and later semantic review.
        """
        pages = self._sanitize_generated_pages(postprocess_story_pages(story_data.get("pages", [])))[:1]
        if not pages:
            return False, "canon_page_1_missing"
        page = pages[0].strip()
        leak_reason = self._canon_instruction_leak_reason(page)
        if leak_reason:
            return False, leak_reason

        language = str(request.storyLanguageCode or "en").strip().lower().replace("_", "-").split("-", 1)[0]
        content_units = self._story_text_units(page, language)
        sentence_count = self._count_story_sentences(page)
        child_age = self._safe_child_age(request.age)
        minimum_units, unit_label = self._first_page_minimum_units(child_age, language)
        minimum_sentences = 3 if child_age <= 5 else 4
        if content_units < minimum_units:
            return False, f"canon_page_1_too_short_{content_units}_{unit_label}"
        if sentence_count < minimum_sentences:
            return False, f"canon_page_1_too_few_sentences_{sentence_count}"

        contract = self._canon_contract(request)
        expected_title = str(contract.get("title") or "").strip()
        returned_title = str(story_data.get("title") or "").strip()
        if expected_title and returned_title != expected_title:
            return False, "canon_page_1_title_mismatch"

        lower = page.lower()
        frame_markers = (
            "settled into bed", "settled in bed", "closed their eyes", "closed his eyes",
            "closed her eyes", "listened to the story", "began to listen", "bedtime story",
            "once upon a time",
        )
        source_terms = self._canon_page_one_terms(request)
        candidate_terms = [term for term in source_terms if len(term.strip()) >= 4]

        # For Japanese and Arabic, prefer anchors written in the requested
        # script. An English fallback catalogue row must not make a correct
        # translated scene impossible to validate locally.
        if language == "ja":
            native_terms = [
                term for term in candidate_terms
                if re.search(r'[\u3040-\u30ff\u3400-\u9fff]', term)
            ]
        elif language == "ar":
            native_terms = [
                term for term in candidate_terms
                if re.search(r'[\u0600-\u06ff]', term)
            ]
        else:
            native_terms = candidate_terms

        terms_to_match = native_terms or candidate_terms
        source_present = any(term.lower() in lower for term in terms_to_match)
        frame_present = any(marker in lower for marker in frame_markers)

        if source_terms and not source_present:
            if language in {"ja", "ar"} and not native_terms:
                # No same-script source anchors exist in the selected catalogue
                # translation. Do not fail a valid translated Canon scene solely
                # because the authoritative record is stored in another script.
                print(
                    f"[PERF] canon_page_1_cross_script_source_guard_fallback "
                    f"lang={language} title={expected_title!r}"
                )
            else:
                return False, "canon_page_1_missing_source_scene"

        if frame_present and content_units < minimum_units + (12 if unit_label == "words" else 18) and not source_present:
            return False, "canon_page_1_only_child_frame"
        return True, "ok"

    def _validate_folk_adventure_first_page(
        self,
        request: GenerateStoryRequest,
        story_data: dict,
    ) -> tuple[bool, str]:
        """Reject a generic Story World Page 1 that lacks source dependency."""
        pages = self._sanitize_generated_pages(postprocess_story_pages(story_data.get("pages", [])))[:1]
        if not pages:
            return False, "folk_adventure_page_1_missing"
        page = pages[0].strip()
        child_age = self._safe_child_age(request.age)
        minimum_words = 28 if child_age <= 5 else 38 if child_age <= 8 else 48
        if len(page.split()) < minimum_words:
            return False, f"folk_adventure_page_1_too_short_{len(page.split())}_words"
        if self._count_story_sentences(page) < (3 if child_age <= 5 else 4):
            return False, "folk_adventure_page_1_too_few_sentences"

        contract = self._folk_adventure_contract(request)
        anchor_terms: list[str] = []
        for value in (
            contract.get("source_title"),
            contract.get("characters"),
            contract.get("locations"),
            contract.get("protected_facts"),
            contract.get("valid_entry_points"),
            contract.get("expandable_consequences"),
        ):
            anchor_terms.extend(self._flatten_canon_text(value))

        lower = page.lower()
        distinctive_terms: list[str] = []
        for raw in anchor_terms:
            cleaned = re.sub(r"\s+", " ", str(raw or "")).strip()
            if not cleaned:
                continue
            if 4 <= len(cleaned) <= 100:
                distinctive_terms.append(cleaned.lower())
            for token in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿĀ-ž’'-]{4,}", cleaned):
                distinctive_terms.append(token.lower().strip("’'-"))

        source_present = any(term and term in lower for term in distinctive_terms[:80])
        if distinctive_terms and not source_present:
            return False, "folk_adventure_page_1_missing_source_dependency"

        return True, "ok"

    async def _generate_first_page_response_with_retry(
        self,
        request: GenerateStoryRequest,
        prompt: str,
        response_schema: dict,
        soft_limit_seconds: float,
    ) -> Dict[str, Any]:
        """Generate and parse Page 1 with one fast retry on malformed JSON.

        Keeps the existing Page-1-first contract: the total time spent here is
        bounded by the existing soft limit. If both attempts fail or time runs
        out, the caller uses the local deterministic Page 1 fallback.
        """
        start = time.time()
        last_error: Optional[Exception] = None
        last_response_text: Optional[str] = None

        for attempt in (1, 2):
            elapsed = time.time() - start
            remaining_timeout = max(0.1, soft_limit_seconds - elapsed)
            if remaining_timeout <= 0.25:
                break

            t_attempt = time.time()
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_content_sync,
                    prompt,
                    response_schema,
                    2048,
                ),
                timeout=remaining_timeout,
            )
            print(
                f"[PERF] first_page Gemini attempt={attempt} "
                f"took={time.time() - t_attempt:.2f}s remaining_budget={max(0, soft_limit_seconds - (time.time() - start)):.2f}s"
            )
            self._log_gemini_response_metadata(f"first_page_attempt_{attempt}", response)

            response_text = getattr(response, 'text', None)
            last_response_text = response_text if isinstance(response_text, str) else None
            if not response_text or not isinstance(response_text, str):
                last_error = ValueError('Failed to generate first page: empty response text')
                self._log_gemini_text_preview(f"first_page_attempt_{attempt}", response_text)
                continue

            try:
                story_data = self._clean_json_response(response_text)
                if not isinstance(story_data, dict) or 'title' not in story_data or 'pages' not in story_data:
                    raise ValueError('Invalid first-page story format returned by AI')
                if self._is_canon_request(request):
                    valid, reason = self._validate_canon_first_page(request, story_data)
                    if not valid:
                        raise ValueError(reason)
                elif self._is_folk_adventure_request(request):
                    valid, reason = self._validate_folk_adventure_first_page(request, story_data)
                    if not valid:
                        raise ValueError(reason)
                return story_data
            except Exception as parse_exc:
                last_error = parse_exc
                self._log_gemini_text_preview(f"first_page_attempt_{attempt}", response_text)
                print(
                    f"[PERF] first_page parse failed attempt={attempt} "
                    f"error={str(parse_exc)[:300]}"
                )
                # One retry only. Do not continue looping beyond attempt 2.
                continue

        if last_response_text:
            raise ValueError(f"First page Gemini JSON failed after retry: {last_error}")
        raise ValueError(f"First page Gemini failed before usable text: {last_error}")

    async def generate_story_first_page(self, request: GenerateStoryRequest, subscription: SubscriptionResponse) -> Dict[str, Any]:
        start_total = time.time()
        print("[PERF] ========================================")
        print(f"[PERF] generate_story_first_page START lang={request.storyLanguageCode} duration={request.durationMin}")

        companion = self._select_companion(request, subscription)
        expected_pages = self._intended_page_count(request)

        if not self.model:
            if self._is_canon_request(request):
                raise HTTPException(
                    status_code=503,
                    detail='Canonical Story Worlds generation is temporarily unavailable',
                )
            page_one = f"Once upon a time, {request.childName} discovered a quiet little path full of wonder. The stars seemed to listen as the bedtime adventure began. With a calm heart, {request.childName} stepped forward to learn something kind about {request.customTheme or self._localized_theme_label(request.theme, request.storyLanguageCode)}."
            pages = postprocess_story_pages([page_one])
            return {
                'title': f"{request.childName}'s Bedtime Adventure",
                'pages': pages[:1],
                'companion': companion,
                'expected_pages': expected_pages,
                'generation_status': 'partial',
                'first_page_generation_source': 'local_standard_fallback',
            }

        try:
            prompt = self._build_first_page_prompt(request, companion)
            print(f"[PERF] first_page prompt chars={len(prompt)}")
            t_gemini = time.time()
            try:
                # Consistency guard: do not let slow or malformed Gemini Page 1
                # output hold the user on the generation screen. Page 1 gets
                # enough JSON output headroom for the JSON structured
                # output path, plus one fast retry if parsing fails. The whole
                # operation remains bounded by FIRST_PAGE_SOFT_LIMIT_SECONDS.
                story_data = await self._generate_first_page_response_with_retry(
                    request=request,
                    prompt=prompt,
                    response_schema=self._story_response_schema(1, include_title=True),
                    soft_limit_seconds=FIRST_PAGE_SOFT_LIMIT_SECONDS,
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - t_gemini
                if self._is_canon_request(request):
                    print(
                        f"[PERF] canon first_page timed out after {elapsed:.2f}s; "
                        "failing closed to prevent unsafe Canon fallback content"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail='Canonical story generation timed out. Please try again.',
                    )
                print(
                    f"[PERF] first_page Gemini soft limit hit after {elapsed:.2f}s; "
                    "using fast fallback page 1"
                )
                fallback = self._build_first_page_fallback(request, companion)
                fallback['generation_fallback_reason'] = 'first_page_timeout'
                fallback['first_page_generation_source'] = 'fallback_timeout'
                fallback_page = (fallback.get('pages') or [''])[0]
                print(f"[PERF] first_page_size fallback words={len(fallback_page.split())} chars={len(fallback_page)}")
                print(f"[PERF] generate_story_first_page DONE fallback total={time.time() - start_total:.2f}s")
                print("[PERF] ========================================")
                return fallback

            elapsed = time.time() - t_gemini
            print(f"[PERF] first_page Gemini completed total={elapsed:.2f}s")

            pages = self._sanitize_generated_pages(postprocess_story_pages(story_data.get('pages', [])))[:1]
            if not pages:
                raise ValueError('First-page story returned no pages')

            if self._is_canon_request(request):
                canon_title = str(self._canon_contract(request).get('title') or story_data.get('title') or '').strip()
                if canon_title:
                    story_data['title'] = canon_title

            page_one_words = len(pages[0].split())
            page_one_chars = len(pages[0])
            print(f"[PERF] first_page_size words={page_one_words} chars={page_one_chars}")
            print(f"[PERF] first_page_ready_for_response pages=1 expected_pages={expected_pages}")
            print(f"[PERF] generate_story_first_page DONE total={time.time() - start_total:.2f}s")
            print("[PERF] ========================================")
            return {
                'title': story_data['title'],
                'pages': pages,
                'companion': companion,
                'expected_pages': expected_pages,
                'generation_status': 'partial',
                'first_page_generation_source': 'gemini_primary',
            }
        except HTTPException:
            raise
        except Exception as exc:
            # Never fall back to full story generation inside the initial
            # Page-1 request. For non-English Canon, fail closed rather than
            # exposing untranslated source material.
            if self._is_canon_request(request):
                print(f"[PERF] canon first_page failed closed: {exc}")
                raise HTTPException(
                    status_code=503,
                    detail='Canonical story generation is temporarily unavailable. Please try again.',
                )
            print(f"[PERF] first_page failed, using deterministic page 1 fallback: {exc}")
            fallback = self._build_first_page_fallback(request, companion)
            fallback['generation_fallback_reason'] = 'first_page_exception'
            fallback['first_page_generation_source'] = 'fallback_exception'
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
                if self._is_canon_request(request):
                    raise RuntimeError("Canonical Story Worlds background generation requires Gemini")
                remaining = [
                    f"On the next part of the path, {request.childName} found a small kindness waiting to be shared.",
                    f"The adventure took one surprising turn, and {request.childName} used the chosen moral to put things right.",
                    f"The problem was solved, but one small sign hinted that another adventure might be waiting for another day.",
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
                    preferred_batch_count = min(
                        BACKGROUND_PAGE_BATCH_SIZE,
                        expected_pages - len(working_pages),
                    )
                    next_page_number = len(working_pages) + 1
                    batch_pages: list[str] = []
                    last_batch_error: Optional[str] = None

                    # Progressive fallback: try the normal 3-page batch first,
                    # then 2 pages, then 1 page. This reduces Gemini truncation
                    # risk on later pages without changing Page-1-first flow.
                    for batch_count in range(preferred_batch_count, 0, -1):
                        try:
                            print(
                                f"[PERF] remaining_pages_attempt story_id={story_id} "
                                f"next_page={next_page_number} count={batch_count}"
                            )
                            batch_pages = await self._generate_remaining_pages_batch(
                                request=request,
                                companion=companion,
                                title=title,
                                working_pages=working_pages,
                                batch_count=batch_count,
                            )
                            print(
                                f"[PERF] remaining_pages_attempt SUCCESS story_id={story_id} "
                                f"next_page={next_page_number} count={batch_count}"
                            )
                            break
                        except Exception as batch_exc:
                            last_batch_error = str(batch_exc)
                            print(
                                f"[PERF] remaining_pages_attempt FAILED story_id={story_id} "
                                f"next_page={next_page_number} count={batch_count} error={last_batch_error[:300]}"
                            )

                    if not batch_pages:
                        # Immediate provider attempts for this exact next page
                        # have been exhausted. A transient HTTP-200/empty-text
                        # Gemini response must not permanently strand a story at
                        # 4/7 (or any other partial count), so keep the existing
                        # playable pages published and perform a small number of
                        # delayed recovery cycles for the SAME next page.
                        #
                        # Each recovery call still uses the existing
                        # _generate_remaining_pages_batch validation/retry logic;
                        # we do not weaken page quality, Canon checks, or invent
                        # fallback story text.
                        safe_pages = postprocess_story_pages(
                            working_pages or current_pages
                        )[:expected_pages]

                        if safe_pages:
                            self._publish_partial_story_pages(
                                story_id=story_id,
                                user_id=user_id,
                                working_pages=safe_pages,
                                expected_pages=expected_pages,
                                generation_error=(
                                    last_batch_error
                                    or "Background continuation did not produce usable pages"
                                ),
                            )

                        for recovery_attempt in range(
                            1,
                            BACKGROUND_CONTINUATION_RECOVERY_ATTEMPTS + 1,
                        ):
                            delay_seconds = (
                                BACKGROUND_CONTINUATION_RECOVERY_DELAY_SECONDS
                                * recovery_attempt
                            )
                            print(
                                f"[PERF] remaining_pages_recovery WAIT story_id={story_id} "
                                f"next_page={next_page_number} "
                                f"attempt={recovery_attempt}/"
                                f"{BACKGROUND_CONTINUATION_RECOVERY_ATTEMPTS} "
                                f"delay={delay_seconds:.2f}s "
                                f"previous_error={(last_batch_error or 'unknown')[:220]}"
                            )
                            await asyncio.sleep(delay_seconds)

                            try:
                                print(
                                    f"[PERF] remaining_pages_recovery TRY story_id={story_id} "
                                    f"next_page={next_page_number} "
                                    f"attempt={recovery_attempt}/"
                                    f"{BACKGROUND_CONTINUATION_RECOVERY_ATTEMPTS}"
                                )
                                batch_pages = await self._generate_remaining_pages_batch(
                                    request=request,
                                    companion=companion,
                                    title=title,
                                    working_pages=working_pages,
                                    batch_count=1,
                                )
                                print(
                                    f"[PERF] remaining_pages_recovery SUCCESS story_id={story_id} "
                                    f"next_page={next_page_number} "
                                    f"attempt={recovery_attempt}/"
                                    f"{BACKGROUND_CONTINUATION_RECOVERY_ATTEMPTS}"
                                )
                                break
                            except Exception as recovery_exc:
                                last_batch_error = str(recovery_exc)
                                print(
                                    f"[PERF] remaining_pages_recovery FAILED story_id={story_id} "
                                    f"next_page={next_page_number} "
                                    f"attempt={recovery_attempt}/"
                                    f"{BACKGROUND_CONTINUATION_RECOVERY_ATTEMPTS} "
                                    f"error={last_batch_error[:300]}"
                                )

                        if not batch_pages:
                            # Preserve the old safe failure mode after bounded
                            # recovery is genuinely exhausted: keep the usable
                            # pages partial rather than marking the story failed.
                            safe_pages = postprocess_story_pages(
                                working_pages or current_pages
                            )[:expected_pages]
                            if safe_pages:
                                self._publish_partial_story_pages(
                                    story_id=story_id,
                                    user_id=user_id,
                                    working_pages=safe_pages,
                                    expected_pages=expected_pages,
                                    generation_error=(
                                        last_batch_error
                                        or "Background continuation recovery exhausted"
                                    ),
                                )
                                print(
                                    f"[PERF] complete_story_background PAUSED_AFTER_RECOVERY "
                                    f"story_id={story_id} "
                                    f"pages={len(safe_pages)}/{expected_pages} "
                                    f"total={time.time() - start_total:.2f}s"
                                )
                                return
                            raise ValueError(
                                last_batch_error
                                or "Background continuation produced no usable pages"
                            )

                    working_pages = postprocess_story_pages([*working_pages, *batch_pages])[:expected_pages]
                    remaining.extend(batch_pages)

                    # Canon uses 7 pages as a provisional maximum. After Page 6,
                    # let the existing semantic Canon reviewer decide whether the
                    # authentic story has already finished. If it has, reduce the
                    # confirmed final page count to the text we actually have and
                    # complete normally. This preserves text polling as the source
                    # of truth and avoids manufacturing a filler Page 7.
                    if (
                        self._is_canon_request(request)
                        and len(working_pages) == expected_pages - 1
                    ):
                        canon_complete, canon_reason = await self._canon_can_finish_on_current_page(
                            request=request,
                            title=title,
                            pages=working_pages,
                        )
                        print(
                            f"[PERF] canon_early_completion_review story_id={story_id} "
                            f"pages={len(working_pages)}/{expected_pages} "
                            f"complete={canon_complete} reason={canon_reason[:220]!r}"
                        )
                        if canon_complete:
                            working_pages[-1] = self._ensure_the_end(working_pages[-1])
                            # Keep `remaining` aligned with the now-final working
                            # pages in case _ensure_the_end changed Page 6 text.
                            remaining = working_pages[len(current_pages):]
                            expected_pages = len(working_pages)
                            print(
                                f"[PERF] canon_story_completed_early story_id={story_id} "
                                f"confirmed_pages={expected_pages}"
                            )
                            break

                    # Publish partial pages immediately. Reader polling can then
                    # advance to pages 2+ without waiting for the full story.
                    if len(working_pages) < expected_pages:
                        self._publish_partial_story_pages(
                            story_id=story_id,
                            user_id=user_id,
                            working_pages=working_pages,
                            expected_pages=expected_pages,
                            generation_error=None,
                        )

            all_pages = postprocess_story_pages([*current_pages, *remaining])[:expected_pages]
            if len(all_pages) < expected_pages:
                safe_pages = postprocess_story_pages(all_pages or current_pages)[:expected_pages]
                if safe_pages:
                    self._publish_partial_story_pages(
                        story_id=story_id,
                        user_id=user_id,
                        working_pages=safe_pages,
                        expected_pages=expected_pages,
                        generation_error=f"Remaining generation produced only {len(safe_pages)} of {expected_pages} pages",
                    )
                    print(
                        f"[PERF] complete_story_background PARTIAL_ONLY story_id={story_id} "
                        f"pages={len(safe_pages)}/{expected_pages} total={time.time() - start_total:.2f}s"
                    )
                    return
                raise ValueError(f"Remaining generation produced only {len(all_pages)} of {expected_pages} pages")

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
                safe_pages = postprocess_story_pages(current_pages)[:expected_pages]
                if safe_pages:
                    self._publish_partial_story_pages(
                        story_id=story_id,
                        user_id=user_id,
                        working_pages=safe_pages,
                        expected_pages=expected_pages,
                        generation_error=str(exc),
                    )
                    return

                self.story_repo.update(story_id, user_id, {
                    'generation_status': 'failed',
                    'expected_pages': expected_pages,
                    'generation_error': str(exc)[:500],
                })
            except Exception as update_exc:
                print(f"[PERF] failed to persist generation failure story_id={story_id}: {update_exc}")

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
        response = self._generate_content_sync(
            prompt,
            self._story_response_schema(7, include_title=True),
        )
        print(f"[PERF] Gemini generate_content took {time.time() - t_gemini:.2f}s")

        response_text = getattr(response, 'text', None)
        if not response_text or not isinstance(response_text, str):
            print(f"[PERF] generate_story FAILED no response text total={time.time() - start_total:.2f}s")
            print("[PERF] ========================================")
            raise HTTPException(status_code=500, detail='Failed to generate story')

        t_clean = time.time()
        print(f"[PERF] cleaning took {time.time() - t_clean:.2f}s response_chars={len(response_text)}")

        t_parse = time.time()
        story_data = self._clean_json_response(response_text)
        print(f"[PERF] JSON parse took {time.time() - t_parse:.2f}s")

        if not isinstance(story_data, dict) or 'title' not in story_data or 'pages' not in story_data:
            print(f"[PERF] generate_story FAILED invalid format total={time.time() - start_total:.2f}s")
            print("[PERF] ========================================")
            raise HTTPException(status_code=500, detail='Invalid story format returned by AI')

        t_post = time.time()
        pages = self._sanitize_generated_pages(postprocess_story_pages(story_data.get('pages', [])))
        print(f"[PERF] postprocess took {time.time() - t_post:.2f}s pages_before_trim={len(pages)}")

        # Hard guard for production performance: Gemini may occasionally exceed
        # the requested page count. Trim to the intended count so narration cost,
        # timing, and reader sync remain predictable.
        intended_page_count = 7
        story_data['pages'] = pages[:intended_page_count]
        if len(story_data['pages']) == intended_page_count:
            story_data['pages'][-1] = self._ensure_the_end(story_data['pages'][-1])
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
                self._generate_content_sync,
                prompt,
                self._metadata_response_schema(),
            )
            print(f"[PERF] extract_metadata Gemini took {time.time() - t_gemini:.2f}s")
            text = getattr(response, 'text', '')
            if not text or not isinstance(text, str):
                print(f"[PERF] extract_metadata empty_response total={time.time() - start_total:.2f}s")
                return {'summary': '', 'characters': [], 'setting': ''}
            result = self._clean_json_response(text)
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

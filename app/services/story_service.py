from __future__ import annotations

import asyncio
import json
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

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
]

# Backward-compatible English seed list kept for any older imports/tests.
OPENING_SEEDS = [seed["en"] for seed in OPENING_SEED_FAMILIES]

FIRST_PAGE_TIMEOUT_SECONDS = 30
# User-facing consistency target: if Gemini has not produced page 1
# quickly enough, return a deterministic page-1 fallback so Reader can open.
# The full story still completes in the normal background Gemini path.
FIRST_PAGE_SOFT_LIMIT_SECONDS = 22


class StoryService:
    def __init__(self, story_repo: StoryRepository):
        self.story_repo = story_repo
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
        self.model = genai.GenerativeModel(settings.gemini_model) if settings.gemini_api_key else None

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

    def _storycraft_rules(self) -> str:
        return """STORYCRAFT QUALITY RULES:
- Make the story feel like a premium illustrated children's fantasy tale: imaginative, emotionally warm, cinematic, and magical, while remaining original and bedtime-safe.
- Use a classic storybook arc: wonder-filled opening, gentle discovery, small emotional challenge, magical or meaningful helper moment, moral learned through action, and a satisfying peaceful resolution.
- The story setting should feel vivid, memorable, and specific, like a real storybook world the child can picture immediately.
- The adventure may begin anywhere that suits the theme, not only near a home, bedroom, window, blanket, or bedtime object.
- Use a wide variety of bedtime-safe locations when appropriate: rainforests, river boats, deserts, castles, islands, mountains, oceans, cloud cities, ancient observatories, magical markets, treehouses, hidden valleys, peaceful pirate harbours, underwater palaces, and faraway lands.
- Make the setting important to the story, not merely background scenery. The place should shape what the child notices, chooses, and gently solves.
- The first page should quickly establish a clear world, a gentle objective, and an emotionally meaningful reason for the child to care about what happens next.
- By the end of page 1, the child should have a clear goal, mystery, challenge, promise, or problem that will drive the rest of the story.
- Avoid stories where the child simply wanders through magical locations without a purpose.
- Do not invent unnecessary physical descriptions of the child such as hair colour, eye colour, skin colour, height, clothing, or appearance unless explicitly provided.
- Focus on the child's role, personality, choices, actions, and connection to the setting instead.
- Let the child make choices, notice details, and grow through the story; do not simply describe events happening around them.
- Avoid repeatedly relying on magical objects as the main story trigger.
- Sometimes begin with a problem, visitor, animal, mystery, missing item, wish, celebration, question, or natural event instead.
- Include sensory storybook details: soft light, gentle sounds, cozy textures, moonlight, stars, nature, kindness, friendship, courage, patience, or wonder where appropriate.
- Every page should have a clear story purpose: discovery, decision, challenge, help, transformation, reflection, or peaceful closure.
- Avoid flat summaries. Write immersive scenes that feel read-aloud, memorable, and emotionally rewarding.
- Keep the mood safe for bedtime: no danger, no frightening villains, no peril, no sadness-heavy ending.
- Do not copy or imitate any existing franchise, character, studio, film, song, or copyrighted story world.
- The moral must drive the story conflict and resolution.
- The child should face a meaningful but gentle problem that can only be solved by demonstrating the chosen moral.
- Do not merely mention the moral or explain it.
- Show the moral through actions, choices, and consequences.
- The story should still work even if the word for the moral never appears.
- Use richer moral variety across stories when appropriate, including kindness, bravery, patience, sharing, empathy, honesty, gratitude, resilience, friendship, teamwork, curiosity, responsibility, forgiveness, generosity, listening, confidence, courage, and calm problem-solving.
- If the chosen moral is broad, make it specific through the story situation. For example: kindness can mean noticing someone left out; patience can mean waiting gently; bravery can mean asking for help or trying again; sharing can mean making room for someone else.
- Avoid turning the moral into a lecture. The child should understand it because the story outcome feels emotionally true.
- Keep the moral age-appropriate, hopeful, and reassuring for bedtime."""

    def _select_opening_seed(self, request: GenerateStoryRequest) -> dict:
        """Select a place-first opening locally before Gemini is called.

        This is intentionally instant: no extra AI call, no database lookup, and
        no narration/playback impact. The opening gives the child a clear place
        to enter before the magical event begins.
        """
        child = request.childName or "the child"
        language_code = (request.storyLanguageCode or "en").lower()[:2]
        seed = random.choice(OPENING_SEED_FAMILIES)
        template = seed.get(language_code) or seed["en"]
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
- Use warm, premium British bedtime storytelling with soft rhythm, clear emotion, and child-friendly magic.
- Keep the story gentle, imaginative, cosy, and easy to read aloud.
"""

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
        target_words = "850-1100"
        max_words_per_page = "175"
        pacing_note = (
            "Create a substantial but calm bedtime story suitable for an approximately eight-minute bedtime experience. "
            "Do not compress the plot into a short summary; let each page include a gentle, memorable story moment."
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

LENGTH AND STRUCTURE REQUIREMENTS (STRICT PERFORMANCE RULES):
- EXACTLY {target_pages} pages. Do not return more or fewer pages.
- EACH page should contain {paragraphs_per_page} gentle paragraphs.
- EACH page should contain approximately {sentence_range} bedtime-friendly sentences in total.
- TOTAL story length MUST be approximately {target_words} words.
- Each page should be substantial, normally 120-165 words, but DO NOT exceed {max_words_per_page} words on any single page.
- Use simple, natural sentences suitable for spoken bedtime narration.
- Do not make pages too short. Avoid summarising scenes in only one or two sentences.
- Every page must move the story forward gently and include one memorable story beat.
- The moral should be discovered through the child's actions, not explained like a lesson.
- The final page must end peacefully and softly.
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
- Do not include notes, markdown, or explanations outside the JSON
- Keep the story calm and readable, but do not make it too short.
- If unsure, prioritise reaching the requested narration length while staying bedtime-safe.
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

    def _build_first_page_prompt(self, request: GenerateStoryRequest, companion: Optional[dict]) -> str:
        blocks = self._language_and_character_blocks(request, companion)

        opening_seed = self._select_opening_seed(request)
        opening = opening_seed["sentence"]
        opening_transition_rule = self._opening_transition_rule(opening_seed["family"])
        language_code = (request.storyLanguageCode or "en").lower()
        is_english = language_code == "en"

        if is_english:
            page_length_rule = "450-525 characters total, including spaces and the opening sentence. Do not exceed 525 characters. If extra description is needed, move it to page 2."
            sentence_rule = "3-5 calm, read-aloud sentences"
            instruction_block = f"""- Continue naturally from the opening above
- Keep the tone warm, magical, calm, and bedtime-safe
- Use one or two soft sensory details only; save richer description for page 2
- Let {request.childName} notice or choose something meaningful
- Do NOT introduce danger, fear, or fast pacing
- Do NOT resolve the story yet"""
        else:
            # Non-English first pages can be slower because the model must obey
            # language-only output while generating valid JSON. Keep the same
            # bedtime shape, but reduce output length and instruction load so
            # page 1 is ready faster. The full 7-page story remains unchanged.
            page_length_rule = "425-500 characters total, including spaces and the opening sentence. Do not exceed 500 characters. If extra description is needed, move it to page 2."
            sentence_rule = "3-5 calm, read-aloud sentences"
            instruction_block = f"""- Continue naturally from the opening above
- Keep the tone warm, magical, calm, and bedtime-safe
- Include one clear, gentle story moment for {request.childName}
- Do NOT introduce danger, fear, or fast pacing
- Do NOT resolve the story yet"""

        return f"""You are continuing a premium children's bedtime story.

IMPORTANT LANGUAGE RULE:
- Write ONLY in {blocks['language_name']}
- Do NOT mix languages
{self._language_style_block(request.storyLanguageCode)}
STORY CONTEXT:
- Child: {request.childName}, age {request.age}
- Theme: {blocks['effective_theme']}
- Moral: {request.moral}
- Calm level: {request.calmLevel}

START THE STORY WITH THIS EXACT SENTENCE:
"{opening}"

Then continue immediately from it.

PLACE-ENTRY RULE:
- {opening_transition_rule}

INSTRUCTIONS:
{instruction_block}

PAGE 1 STRUCTURE:
- {page_length_rule}
- This length limit is important because page 1 starts narration. Keep page 1 short, clear, and complete; move extra world-building to page 2.
- 1-2 gentle paragraphs
- {sentence_rule}
- The first page MUST begin like a classic children's story.
- First establish:
  - who {request.childName} is in this story world,
  - where they live or where they begin the story,
  - and one or two simple details about their normal world.
- If the opening sentence has already introduced {request.childName} naturally, do not introduce them again.
- Do not repeat the child's name as a second introduction immediately after the opening sentence.
- Expand naturally from the opening sentence by describing:
  - who the child is in this story world,
  - what their normal life is like,
  - and why this setting matters.
- If a role or identity fits naturally, weave it into the existing setup rather than starting a new introduction.
  Good example: "{request.childName}, a curious young explorer, loved searching the shore for treasures."
  Avoid: "{request.childName} lived on the island. A curious young explorer named {request.childName}..."
- Only then introduce:
  - the magical discovery,
  - mystery,
  - wish,
  - challenge,
  - or gentle adventure.
- Think:
  "Once upon a time there lived..."
  before
  "And then something happened..."
- The reader should understand why {request.childName} is in this setting before the adventure begins.
- By the end of page 1, the child must understand:
  - who the main character is,
  - where they are,
  - what has changed,
  - and what needs to happen next.
- By the end of page 1, introduce a clear story promise.
- The child should understand what they are trying to achieve, discover, solve, help, find, or learn during the adventure.
- Avoid openings that jump straight to a magical object, door, clue, or event before explaining who {request.childName} is and why they are there.
- Do not add clothing or appearance details unless the parent provided them.
- Keep the setup magical, warm, bedtime-safe, and easy for a child to follow.

COMPANION:
- {blocks['companion_line']}

CHARACTERS:
- {blocks['character_instruction']}

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON:
{{"title":"Short magical title","pages":["page 1 text"]}}
"""

    def _build_first_page_fallback(self, request: GenerateStoryRequest, companion: Optional[dict]) -> Dict[str, Any]:
        """Fast varied page-1 fallback used only when Gemini is too slow.

        This protects the launch UX from occasional LLM latency spikes without
        repeatedly returning the same story. The remaining pages still complete
        through Gemini in the normal background flow, so full story quality is
        preserved after the reader opens.
        """
        child = request.childName or "the child"
        language_code = (request.storyLanguageCode or "en").lower()[:2]
        expected_pages = self._intended_page_count(request)
        theme = self._localized_theme_label(request.theme, language_code) or "magic"
        moral = str(request.moral or "kindness").strip().lower()

        opening_seed = self._select_opening_seed(request)
        opening = opening_seed.get("sentence") or f"Once upon a time, {child} discovered a quiet little path full of wonder."

        fallback_variants = {
            "en": [
                {
                    "title": f"{child} and the Moonlit Map",
                    "middle": (
                        f"That evening, {child} noticed a small silver map folded beside a sleepy lantern. "
                        f"The map did not rush or sparkle loudly; it simply waited, as if teaching {child} that {moral} sometimes begins with one quiet choice. "
                        f"With a calm breath, {child} followed the first gentle clue and wondered what soft magic the night might share."
                    ),
                },
                {
                    "title": f"{child} and the Whispering Lantern",
                    "middle": (
                        f"As the evening grew still, a little lantern began to glow with a warm honey light. "
                        f"It seemed to whisper that a small bedtime journey about {theme} was waiting nearby. "
                        f"{child} listened carefully, remembering that {moral} could help even the smallest light shine brighter."
                    ),
                },
                {
                    "title": f"{child} and the Sleepy Moon Garden",
                    "middle": (
                        f"Near the quietest corner, {child} found a path sprinkled with pale moon-dust and tiny sleeping flowers. "
                        f"Nothing hurried there; every leaf seemed to breathe slowly in the night air. "
                        f"{child} stepped forward gently, ready to learn how {moral} could help the garden wake just enough to share its secret."
                    ),
                },
                {
                    "title": f"{child} and the Little Cloud Boat",
                    "middle": (
                        f"A tiny cloud boat drifted close, rocking softly as if it had been waiting for a careful passenger. "
                        f"Inside was a folded note asking for help with a gentle {theme} journey before the stars settled down. "
                        f"{child} climbed in quietly, knowing that {moral} would matter more than rushing ahead."
                    ),
                },
            ],
            "es": [
                {
                    "title": f"{child} y el mapa de luna",
                    "middle": (
                        f"Aquella tarde, {child} encontró un pequeño mapa plateado junto a una linterna tranquila. "
                        f"El mapa no tenía prisa; parecía recordar que {moral} podía empezar con una decisión pequeña y amable. "
                        f"Con un suspiro sereno, {child} siguió la primera pista y se preguntó qué magia suave guardaba la noche."
                    ),
                },
                {
                    "title": f"{child} y la linterna susurrante",
                    "middle": (
                        f"Cuando todo quedó en calma, una linterna empezó a brillar con una luz cálida. "
                        f"Parecía anunciar un pequeño cuento de {theme}, tranquilo y seguro. "
                        f"{child} escuchó con atención, recordando que {moral} podía ayudar incluso a la luz más pequeña."
                    ),
                },
            ],
            "fr": [
                {
                    "title": f"{child} et la carte de lune",
                    "middle": (
                        f"Ce soir-là, {child} aperçut une petite carte argentée près d’une lanterne calme. "
                        f"La carte ne pressait personne; elle semblait rappeler que {moral} commence parfois par un tout petit choix doux. "
                        f"Avec une respiration tranquille, {child} suivit le premier indice et se demanda quelle magie tendre attendait dans la nuit."
                    ),
                },
                {
                    "title": f"{child} et la lanterne qui murmurait",
                    "middle": (
                        f"Quand le soir devint silencieux, une petite lanterne se mit à briller d’une lumière chaude. "
                        f"Elle semblait annoncer une douce aventure de {theme}, calme et rassurante. "
                        f"{child} écouta avec attention, en se souvenant que {moral} pouvait aider même la plus petite lumière."
                    ),
                },
            ],
            "de": [
                {
                    "title": f"{child} und die Mondkarte",
                    "middle": (
                        f"An diesem Abend entdeckte {child} neben einer stillen Laterne eine kleine silberne Karte. "
                        f"Die Karte hatte keine Eile; sie erinnerte leise daran, dass {moral} manchmal mit einer kleinen freundlichen Entscheidung beginnt. "
                        f"Mit einem ruhigen Atemzug folgte {child} dem ersten Hinweis und fragte sich, welche sanfte Magie die Nacht bereithielt."
                    ),
                },
                {
                    "title": f"{child} und die flüsternde Laterne",
                    "middle": (
                        f"Als der Abend ganz still wurde, begann eine kleine Laterne warm zu leuchten. "
                        f"Sie schien von einem ruhigen Abenteuer über {theme} zu erzählen. "
                        f"{child} hörte aufmerksam zu und dachte daran, dass {moral} selbst dem kleinsten Licht helfen konnte."
                    ),
                },
            ],
            "it": [
                {
                    "title": f"{child} e la mappa di luna",
                    "middle": (
                        f"Quella sera, {child} trovò una piccola mappa d’argento accanto a una lanterna tranquilla. "
                        f"La mappa non aveva fretta; sembrava ricordare che {moral} a volte comincia con una piccola scelta gentile. "
                        f"Con un respiro calmo, {child} seguì il primo indizio e si chiese quale magia dolce aspettasse nella notte."
                    ),
                },
                {
                    "title": f"{child} e la lanterna che sussurrava",
                    "middle": (
                        f"Quando la sera diventò silenziosa, una piccola lanterna iniziò a brillare di luce calda. "
                        f"Sembrava annunciare una tenera avventura di {theme}, calma e sicura. "
                        f"{child} ascoltò con attenzione, ricordando che {moral} poteva aiutare anche la luce più piccola."
                    ),
                },
            ],
        }

        variants = fallback_variants.get(language_code, fallback_variants["en"])
        selected = random.choice(variants)
        page = f"{opening} {selected['middle']}"
        pages = postprocess_story_pages([page])[:1]
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
        page_one: str,
        remaining_page_count: int,
    ) -> str:
        blocks = self._language_and_character_blocks(request, companion)
        return f"""You are continuing a premium children's bedtime story.

IMPORTANT LANGUAGE RULE:
- Continue ONLY in {blocks['language_name']}.
- Do NOT use English unless the language is English.
- Do NOT mix languages.
{self._language_style_block(request.storyLanguageCode)}
ORIGINAL STORY REQUIREMENTS:
- Child name: {request.childName}
- Age: {request.age}
- Theme: {blocks['effective_theme']}
- Moral: {request.moral}
- Calm level: {request.calmLevel}
- Tone: warm, magical, calming, bedtime-safe
- Use simple, natural language suitable for reading aloud
- No scary content
- No rushed ending
- Do NOT write "The end"
- End peacefully and softly on the final page.
- The complete story should feel suitable for an approximately eight-minute bedtime experience.

{self._storycraft_rules()}

EXISTING STORY START:
Title: {title}
Page 1: {page_one}

CONTINUATION REQUIREMENTS:
- Write exactly {remaining_page_count} remaining pages.
- Continue naturally from page 1.
- Do not recap page 1.
- Do not contradict page 1.
- Each page should contain 2-3 gentle paragraphs.
- Each page should be approximately 120-170 words.
- Each page should contain around 5-7 bedtime-friendly sentences in total.
- Do not make pages too short; each page should feel like a complete story moment, not a summary.
- Every page must move the story forward gently and include one memorable story beat.
- The moral should be discovered through the child's actions, not explained like a lesson.
- The final page must end peacefully and softly.

COMPANION:
- {blocks['companion_line']}

CHARACTERS:
- {blocks['character_instruction']}

OUTPUT FORMAT (STRICT):
Return ONLY valid JSON:
{{"pages": ["page 2 text", "page 3 text"]}}

OUTPUT QUALITY RULES:
- Return continuation pages only.
- Do not include notes, markdown, or explanations outside the JSON.
- The JSON pages array must contain exactly {remaining_page_count} strings."""

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

        try:
            prompt = self._build_first_page_prompt(request, companion)
            print(f"[PERF] first_page prompt chars={len(prompt)}")
            t_gemini = time.time()
            try:
                # Consistency guard: do not let a slow Gemini first-page call hold
                # the user on the generation screen. If page 1 is not back within
                # the soft limit, return a polished deterministic page 1 and let
                # the remaining story continue through the normal background path.
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.model.generate_content, prompt),
                    timeout=FIRST_PAGE_SOFT_LIMIT_SECONDS,
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - t_gemini
                print(
                    f"[PERF] first_page Gemini soft limit hit after {elapsed:.2f}s; "
                    "using fast fallback page 1"
                )
                fallback = self._build_first_page_fallback(request, companion)
                fallback_page = (fallback.get('pages') or [''])[0]
                print(f"[PERF] first_page_size fallback words={len(fallback_page.split())} chars={len(fallback_page)}")
                print(f"[PERF] generate_story_first_page DONE fallback total={time.time() - start_total:.2f}s")
                print("[PERF] ========================================")
                return fallback

            elapsed = time.time() - t_gemini
            print(f"[PERF] first_page Gemini took {elapsed:.2f}s")

            response_text = getattr(response, 'text', None)
            if not response_text or not isinstance(response_text, str):
                raise ValueError('Failed to generate first page')

            story_data = self._clean_json_response(response_text)
            if not isinstance(story_data, dict) or 'title' not in story_data or 'pages' not in story_data:
                raise ValueError('Invalid first-page story format returned by AI')

            pages = postprocess_story_pages(story_data.get('pages', []))[:1]
            if not pages:
                raise ValueError('First-page story returned no pages')

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
            }
        except Exception as exc:
            print(f"[PERF] first_page failed, falling back to full story: {exc}")
            full_story = await self.generate_story(request, subscription)
            full_story['expected_pages'] = len(full_story.get('pages') or [])
            full_story['generation_status'] = 'complete'
            return full_story

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

                prompt = self._build_remaining_pages_prompt(
                    request=request,
                    companion=companion,
                    title=title,
                    page_one=current_pages[0] if current_pages else '',
                    remaining_page_count=remaining_count,
                )
                print(f"[PERF] remaining_pages prompt chars={len(prompt)} expected_remaining={remaining_count}")
                t_gemini = time.time()
                response = await asyncio.to_thread(self.model.generate_content, prompt)
                print(f"[PERF] remaining_pages Gemini took {time.time() - t_gemini:.2f}s")

                response_text = getattr(response, 'text', None)
                if not response_text or not isinstance(response_text, str):
                    raise ValueError('Failed to generate remaining pages')

                story_data = self._clean_json_response(response_text)
                if not isinstance(story_data, dict) or 'pages' not in story_data:
                    raise ValueError('Invalid remaining-pages story format returned by AI')

                remaining = postprocess_story_pages(story_data.get('pages', []))

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

            t_metadata = time.time()
            print(f"[PERF] metadata_extract START story_id={story_id}")
            metadata = await self.extract_metadata(title, full_text)
            print(f"[PERF] metadata_extract DONE story_id={story_id} total={time.time() - t_metadata:.2f}s")
            update_payload.update({
                'story_summary': metadata.get('summary', ''),
                'characters': metadata.get('characters', []),
                'setting': metadata.get('setting', ''),
            })

            t_update = time.time()
            print(f"[PERF] story_update_complete START story_id={story_id}")
            self.story_repo.update(story_id, user_id, update_payload)
            print(f"[PERF] story_update_complete DONE story_id={story_id} total={time.time() - t_update:.2f}s")
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
        response = self.model.generate_content(prompt)
        print(f"[PERF] Gemini generate_content took {time.time() - t_gemini:.2f}s")

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
        story_data = json.loads(cleaned.strip())
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
            response = self.model.generate_content(prompt)
            print(f"[PERF] extract_metadata Gemini took {time.time() - t_gemini:.2f}s")
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

# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from pathlib import Path
from arduino.app_internal.core.ei import EdgeImpulseRunnerFacade, normalize_ei_classification
from arduino.app_utils import HttpClient


@pytest.fixture(autouse=True)
def mock_infra(monkeypatch: pytest.MonkeyPatch):
    """Mock the infrastructure for Edge Impulse tests.

    This fixture sets up a fake docker-compose configuration to avoid
    real docker-compose lookups and network calls during the tests.

    It also mocks the `get_image_bytes` function to return the input
    bytes unchanged.
    """
    # avoid real docker-compose lookups
    fake = {"services": {"ei-inference": {"ports": ["${BIND_ADDRESS:-127.0.0.1}:1337:1337"]}}}
    monkeypatch.setattr("arduino.app_internal.core.ei.load_brick_compose_file", lambda cls: fake)
    monkeypatch.setattr("arduino.app_internal.core.resolve_address", lambda h: "127.0.0.1")
    monkeypatch.setattr("arduino.app_internal.core.parse_docker_compose_variable", lambda s: [(None, None), (None, "1337")])
    # identity for get_image_bytes
    monkeypatch.setattr("arduino.app_utils.image.get_image_bytes", lambda b: b)


@pytest.fixture
def facade():
    """Fixture for the EdgeImpulseRunnerFacade class."""
    return EdgeImpulseRunnerFacade()


def test_infer_from_file_empty(tmp_path: Path, facade: EdgeImpulseRunnerFacade):
    """Test the infer_from_file method with an empty file path.

    Args:
        tmp_path (Path): A temporary directory path.
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
    """
    assert facade.infer_from_file("") is None


def test_infer_from_file_delegates(tmp_path: Path, facade: EdgeImpulseRunnerFacade, monkeypatch: pytest.MonkeyPatch):
    """Test the infer_from_file method with a valid file path.

    Args:
        tmp_path (Path): A temporary directory path.
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
    """
    f = tmp_path / "x.jpg"
    f.write_bytes(b"data")
    monkeypatch.setattr(facade, "infer_from_image", lambda image_bytes, image_type: {"ok": True})
    out = facade.infer_from_file(str(f))
    assert out == {"ok": True}


def test_infer_invalid_inputs(facade: EdgeImpulseRunnerFacade):
    """Test the infer method with invalid inputs.

    This test checks the behavior of the infer method when provided with
    invalid image data or image type. It ensures that the method returns
    None for invalid inputs, such as empty byte strings or unsupported
    image types.

    Args:
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
    """
    assert facade.infer_from_image(b"", "jpg") is None
    assert facade.infer_from_image(b"data", "") is None
    assert facade.infer_from_image(b"data", "bmp") is None


def test_infer_success_and_error(monkeypatch: pytest.MonkeyPatch, facade: EdgeImpulseRunnerFacade):
    """Test the infer method for success and error cases.

    This test checks the behavior of the infer method when provided with
    valid image data and image type. It also tests the handling of HTTP
    errors and exceptions during the request.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
    """
    # success 200
    seen = {}

    class Resp:
        status_code = 200

        def json(self):
            return {"foo": 1}

    def fake_post(url, files=None):  # noqa
        seen["url"] = url
        seen["files"] = files
        return Resp()

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)
    out = facade.infer_from_image(b"data", "jpg")
    assert out == {"foo": 1}
    assert seen["url"].endswith(":1337/api/image")

    # http error
    class Bad:
        status_code = 500
        text = "err"

    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", lambda *a, **k: Bad())
    assert facade.infer_from_image(b"data", "png") is None
    # exception
    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert facade.infer_from_image(b"data", "png") is None


def test_process_various(facade: EdgeImpulseRunnerFacade, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test the process method with various input types.

    Args:
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
        tmp_path (Path): A temporary directory path.
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
    """
    # string path
    f = tmp_path / "a.png"
    f.write_bytes(b"x")
    monkeypatch.setattr(facade, "infer_from_image", lambda b, t: {"ok": 1})
    assert facade.process(str(f)) == {"ok": 1}
    # dict with image
    item = {"image": b"x", "image_type": "png"}
    monkeypatch.setattr(facade, "infer_from_image", lambda b, t: {"y": 2})
    assert facade.process(item) == {"y": 2}
    # dict missing image => passthrough
    junk = {"foo": "bar"}
    assert facade.process(junk) == junk
    # other types => passthrough
    assert facade.process(123) == 123


def test_get_model_info(monkeypatch: pytest.MonkeyPatch, facade: EdgeImpulseRunnerFacade):
    """Test the get_model_info method of EdgeImpulseRunnerFacade.

    This test checks if the method correctly constructs the URL and retrieves
    model information from the Edge Impulse service.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
    """

    class FakeResp:
        status_code = 200

        def json(self):
            return {
                "project": {
                    "deploy_version": 19,
                    "id": 2251,
                    "impulse_id": 1,
                    "name": "test_model",
                    "version": "1.0.0",
                    "description": "Test model for Edge Impulse",
                    "owner": "Jan Inc.",
                },
                "modelParameters": {
                    "frequency": 16000,
                    "input_features_count": 128,
                    "label_count": 3,
                    "image_input_height": 320,
                    "image_input_width": 320,
                    "labels": ["label1", "label2", "label3"],
                    "model_type": "object_detection",
                    "sensor": 3,
                    "slice_size": 25600,
                    "thresholds": [{"id": 6, "min_score": 0.4000000059604645, "type": "object_detection"}],
                },
            }

    captured = {}

    def fake_get(
        self,
        url: str,
        method: str = "GET",
        data: dict | str = None,
        json: dict = None,
        headers: dict = None,
        timeout: int = 5,
    ):
        captured["url"] = url
        return FakeResp()

    # Mock the requests.get method to return a fake response
    monkeypatch.setattr(HttpClient, "request_with_retry", fake_get)

    info = facade.get_model_info()
    assert captured["url"].endswith("/api/info")
    assert info.name == "test_model"
    assert info.input_features_count == 128
    assert info.label_count == 3
    assert info.frequency == 16000
    assert info.labels == ["label1", "label2", "label3"]
    assert info.model_type == "object_detection"
    assert info.thresholds is not None
    assert isinstance(info.thresholds, list)
    assert info.thresholds[0]["id"] == 6 and info.thresholds[0]["min_score"] == 0.4000000059604645


def test_infer_from_features(monkeypatch: pytest.MonkeyPatch, facade: EdgeImpulseRunnerFacade):
    """Test the infer_from_features method of EdgeImpulseRunnerFacade.

    This test checks if the method correctly sends features to the Edge Impulse service
    and retrieves the inference results.

    Args:
        monkeypatch (pytest.MonkeyPatch): A pytest fixture for monkeypatching.
        facade (EdgeImpulseRunnerFacade): An instance of the EdgeImpulseRunnerFacade class.
    """
    features = [0.1, 0.2, 0.3]
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"result": "success"}

    def fake_post(url: str, json: dict):
        captured["url"] = url
        captured["json"] = json
        return FakeResp()

    # Mock the requests.post method to return a fake response
    monkeypatch.setattr("arduino.app_internal.core.ei.requests.post", fake_post)

    class FakeResp2:
        status_code = 200

        def json(self):
            return {
                "project": {
                    "deploy_version": 163,
                    "id": 412593,
                    "impulse_id": 1,
                    "impulse_name": "Impulse #1",
                    "name": "Tutorial: Continuous motion recognition",
                    "owner": "Edge Impulse Inc.",
                },
                "modelParameters": {
                    "has_visual_anomaly_detection": False,
                    "axis_count": 3,
                    "frequency": 62.5,
                    "has_anomaly": 1,
                    "has_object_tracking": False,
                    "image_channel_count": 0,
                    "image_input_frames": 0,
                    "image_input_height": 0,
                    "image_input_width": 0,
                    "image_resize_mode": "none",
                    "inferencing_engine": 4,
                    "input_features_count": 375,
                    "interval_ms": 16,
                    "label_count": 4,
                    "labels": ["idle", "snake", "updown", "wave"],
                    "model_type": "classification",
                    "sensor": 2,
                    "slice_size": 31,
                    "thresholds": [],
                    "use_continuous_mode": False,
                    "sensorType": "accelerometer",
                },
            }

    def fake_get(
        self,
        url: str,
        method: str = "GET",
        data: dict | str = None,
        json: dict = None,
        headers: dict = None,
        timeout: int = 5,
    ):
        return FakeResp2()

    # Mock the requests.get method to return a fake response
    monkeypatch.setattr(HttpClient, "request_with_retry", fake_get)

    # Mock docker-compose related functions
    fake_compose = {"services": {"ei-inference": {"ports": ["${BIND_ADDRESS:-127.0.0.1}:1337:1337"]}}}
    monkeypatch.setattr("arduino.app_internal.core.ei.load_brick_compose_file", lambda cls: fake_compose)
    monkeypatch.setattr("arduino.app_internal.core.resolve_address", lambda h: "127.0.0.1")
    monkeypatch.setattr("arduino.app_internal.core.parse_docker_compose_variable", lambda s: [(None, None), (None, "1337")])

    result = facade.infer_from_features(features)
    assert captured["url"].endswith("/api/features")
    assert captured["json"] == {"features": features}
    assert result == {"result": "success"}


def test_normalize_ei_classification():
    """Test normalization of Edge Impulse classification results.

    Uses a real-world classification dict with ~1000 classes, most at 0 confidence,
    and verifies that renormalizing the non-zero values produces meaningful percentages
    while preserving the ranking of the top classes.
    """
    det_classifications = {
        'Afghan hound': 0, 'African chameleon': 0, 'African crocodile': 0, 'African elephant': 0,
        'African grey': 0, 'African hunting dog': 0, 'Airedale': 0, 'American Staffordshire terrier': 0,
        'American alligator': 0, 'American black bear': 0, 'American chameleon': 0, 'American coot': 0,
        'American egret': 0, 'American lobster': 0, 'Angora': 0, 'Appenzeller': 0, 'Arabian camel': 0,
        'Arctic fox': 0, 'Australian terrier': 0, 'Band Aid': 0, 'Bedlington terrier': 0,
        'Bernese mountain dog': 0, 'Blenheim spaniel': 0, 'Border collie': 0, 'Border terrier': 0,
        'Boston bull': 0, 'Bouvier des Flandres': 0, 'Brabancon griffon': 0, 'Brittany spaniel': 0,
        'CD player': 0, 'Cardigan': 0, 'Chesapeake Bay retriever': 0, 'Chihuahua': 0,
        'Christmas stocking': 0, 'Crock Pot': 0, 'Dandie Dinmont': 0, 'Doberman': 0,
        'Dungeness crab': 0, 'Dutch oven': 0, 'Egyptian cat': 0, 'English foxhound': 0,
        'English setter': 0, 'English springer': 0, 'EntleBucher': 0, 'Eskimo dog': 0,
        'European fire salamander': 0, 'European gallinule': 0, 'French bulldog': 0, 'French horn': 0,
        'French loaf': 0, 'German shepherd': 0, 'German short-haired pointer': 0, 'Gila monster': 0,
        'Gordon setter': 0, 'Granny Smith': 0, 'Great Dane': 0, 'Great Pyrenees': 0,
        'Greater Swiss Mountain dog': 0, 'Ibizan hound': 0, 'Indian cobra': 0, 'Indian elephant': 0,
        'Irish setter': 0, 'Irish terrier': 0, 'Irish water spaniel': 0, 'Irish wolfhound': 0,
        'Italian greyhound': 0, 'Japanese spaniel': 0, 'Kerry blue terrier': 0, 'Komodo dragon': 0,
        'Labrador retriever': 0, 'Lakeland terrier': 0, 'Leonberg': 0, 'Lhasa': 0, 'Loafer': 0,
        'Madagascar cat': 0, 'Maltese dog': 0, 'Mexican hairless': 0, 'Model T': 0, 'Newfoundland': 0,
        'Norfolk terrier': 0, 'Norwegian elkhound': 0, 'Norwich terrier': 0, 'Old English sheepdog': 0,
        'Pekinese': 0, 'Pembroke': 0, 'Persian cat': 0, 'Petri dish': 0, 'Polaroid camera': 0,
        'Pomeranian': 0, 'Rhodesian ridgeback': 0, 'Rottweiler': 0, 'Saint Bernard': 0, 'Saluki': 0,
        'Samoyed': 0, 'Scotch terrier': 0, 'Scottish deerhound': 0, 'Sealyham terrier': 0,
        'Shetland sheepdog': 0, 'Shih-Tzu': 0, 'Siamese cat': 0, 'Siberian husky': 0,
        'Staffordshire bullterrier': 0, 'Sussex spaniel': 0, 'Tibetan mastiff': 0, 'Tibetan terrier': 0,
        'Walker hound': 0, 'Weimaraner': 0, 'Welsh springer spaniel': 0,
        'West Highland white terrier': 0, 'Windsor tie': 0, 'Yorkshire terrier': 0, 'abacus': 0,
        'abaya': 0, 'academic gown': 0, 'accordion': 0, 'acorn': 0, 'acorn squash': 0,
        'acoustic guitar': 0, 'admiral': 0, 'affenpinscher': 0, 'agama': 0, 'agaric': 0,
        'aircraft carrier': 0, 'airliner': 0, 'airship': 0, 'albatross': 0, 'alligator lizard': 0,
        'alp': 0, 'altar': 0, 'ambulance': 0, 'amphibian': 0, 'analog clock': 0, 'anemone fish': 0,
        'ant': 0, 'apiary': 0, 'apron': 0, 'armadillo': 0, 'artichoke': 0, 'ashcan': 0,
        'assault rifle': 0, 'axolotl': 0, 'baboon': 0, 'backpack': 0, 'badger': 0, 'bagel': 0,
        'bakery': 0, 'balance beam': 0, 'bald eagle': 0, 'balloon': 0, 'ballplayer': 0,
        'ballpoint': 0, 'banana': 0, 'banded gecko': 0, 'banjo': 0, 'bannister': 0, 'barbell': 0,
        'barber chair': 0, 'barbershop': 0, 'barn': 0, 'barn spider': 0, 'barometer': 0,
        'barracouta': 0, 'barrel': 0, 'barrow': 0, 'baseball': 0, 'basenji': 0, 'basketball': 0,
        'basset': 0, 'bassinet': 0, 'bassoon': 0, 'bath towel': 0,
        'bathing cap': 0.015897653996944427, 'bathtub': 0, 'beach wagon': 0, 'beacon': 0,
        'beagle': 0, 'beaker': 0, 'bearskin': 0, 'beaver': 0, 'bee': 0, 'bee eater': 0,
        'beer bottle': 0, 'beer glass': 0, 'bell cote': 0, 'bell pepper': 0, 'bib': 0,
        'bicycle-built-for-two': 0, 'bighorn': 0, 'bikini': 0, 'binder': 0, 'binoculars': 0,
        'birdhouse': 0, 'bison': 0, 'bittern': 0, 'black and gold garden spider': 0,
        'black grouse': 0, 'black stork': 0, 'black swan': 0, 'black widow': 0,
        'black-and-tan coonhound': 0, 'black-footed ferret': 0, 'bloodhound': 0, 'bluetick': 0,
        'boa constrictor': 0, 'boathouse': 0, 'bobsled': 0, 'bolete': 0, 'bolo tie': 0,
        'bonnet': 0, 'book jacket': 0, 'bookcase': 0, 'bookshop': 0, 'borzoi': 0, 'bottlecap': 0,
        'bow': 0, 'bow tie': 0, 'box turtle': 0, 'boxer': 0, 'brain coral': 0, 'brambling': 0,
        'brass': 0, 'brassiere': 0, 'breakwater': 0, 'breastplate': 0, 'briard': 0, 'broccoli': 0,
        'broom': 0, 'brown bear': 0, 'bubble': 0, 'bucket': 0, 'buckeye': 0, 'buckle': 0,
        'bulbul': 0, 'bull mastiff': 0, 'bullet train': 0, 'bulletproof vest': 0, 'bullfrog': 0,
        'burrito': 0, 'bustard': 0, 'butcher shop': 0, 'butternut squash': 0, 'cab': 0,
        'cabbage butterfly': 0, 'cairn': 0, 'caldron': 0, 'can opener': 0, 'candle': 0,
        'cannon': 0, 'canoe': 0, 'capuchin': 0, 'car mirror': 0, 'car wheel': 0, 'carbonara': 0,
        'cardigan': 0, 'cardoon': 0, 'carousel': 0, "carpenter's kit": 0, 'carton': 0,
        'cash machine': 0, 'cassette': 0, 'cassette player': 0, 'castle': 0, 'catamaran': 0,
        'cauliflower': 0, 'cello': 0, 'cellular telephone': 0, 'centipede': 0, 'chain': 0,
        'chain mail': 0, 'chain saw': 0, 'chainlink fence': 0, 'chambered nautilus': 0,
        'cheeseburger': 0, 'cheetah': 0, 'chest': 0, 'chickadee': 0, 'chiffonier': 0, 'chime': 0,
        'chimpanzee': 0, 'china cabinet': 0, 'chiton': 0, 'chocolate sauce': 0, 'chow': 0,
        'church': 0, 'cicada': 0, 'cinema': 0, 'cleaver': 0, 'cliff': 0, 'cliff dwelling': 0,
        'cloak': 0, 'clog': 0, 'clumber': 0, 'cock': 0, 'cocker spaniel': 0, 'cockroach': 0,
        'cocktail shaker': 0, 'coffee mug': 0, 'coffeepot': 0, 'coho': 0, 'coil': 0, 'collie': 0,
        'colobus': 0, 'combination lock': 0, 'comic book': 0, 'common iguana': 0, 'common newt': 0,
        'computer keyboard': 0, 'conch': 0, 'confectionery': 0, 'consomme': 0, 'container ship': 0,
        'convertible': 0, 'coral fungus': 0, 'coral reef': 0, 'corkscrew': 0, 'corn': 0,
        'cornet': 0, 'coucal': 0, 'cougar': 0, 'cowboy boot': 0, 'cowboy hat': 0, 'coyote': 0,
        'cradle': 0, 'crane': 0, 'crash helmet': 0, 'crate': 0.023758962750434875, 'crayfish': 0,
        'crib': 0, 'cricket': 0, 'croquet ball': 0, 'crossword puzzle': 0, 'crutch': 0,
        'cucumber': 0, 'cuirass': 0, 'cup': 0, 'curly-coated retriever': 0, 'custard apple': 0,
        'daisy': 0, 'dalmatian': 0, 'dam': 0, 'damselfly': 0, 'desk': 0, 'desktop computer': 0,
        'dhole': 0, 'dial telephone': 0, 'diamondback': 0, 'diaper': 0, 'digital clock': 0,
        'digital watch': 0, 'dingo': 0, 'dining table': 0, 'dishrag': 0, 'dishwasher': 0,
        'disk brake': 0, 'dock': 0, 'dogsled': 0, 'dome': 0, 'doormat': 0, 'dough': 0,
        'dowitcher': 0, 'dragonfly': 0, 'drake': 0, 'drilling platform': 0, 'drum': 0,
        'drumstick': 0, 'dugong': 0, 'dumbbell': 0, 'dung beetle': 0, 'ear': 0, 'earthstar': 0,
        'echidna': 0, 'eel': 0, 'eft': 0, 'eggnog': 0, 'electric fan': 0, 'electric guitar': 0,
        'electric locomotive': 0, 'electric ray': 0, 'entertainment center': 0, 'envelope': 0,
        'espresso': 0, 'espresso maker': 0, 'face powder': 0, 'feather boa': 0, 'fiddler crab': 0,
        'fig': 0, 'file': 0, 'fire engine': 0, 'fire screen': 0, 'fireboat': 0, 'flagpole': 0,
        'flamingo': 0, 'flat-coated retriever': 0, 'flatworm': 0, 'flute': 0, 'fly': 0,
        'folding chair': 0, 'football helmet': 0, 'forklift': 0, 'fountain': 0, 'fountain pen': 0,
        'four-poster': 0, 'fox squirrel': 0, 'freight car': 0, 'frilled lizard': 0, 'frying pan': 0,
        'fur coat': 0, 'gar': 0, 'garbage truck': 0, 'garden spider': 0, 'garter snake': 0,
        'gas pump': 0, 'gasmask': 0, 'gazelle': 0, 'geyser': 0, 'giant panda': 0,
        'giant schnauzer': 0, 'gibbon': 0, 'go-kart': 0, 'goblet': 0, 'golden retriever': 0,
        'goldfinch': 0, 'goldfish': 0, 'golf ball': 0, 'golfcart': 0, 'gondola': 0, 'gong': 0,
        'goose': 0, 'gorilla': 0, 'gown': 0, 'grand piano': 0, 'grasshopper': 0,
        'great grey owl': 0, 'great white shark': 0, 'green lizard': 0, 'green mamba': 0,
        'green snake': 0, 'greenhouse': 0, 'grey fox': 0, 'grey whale': 0, 'grille': 0,
        'grocery store': 0, 'groenendael': 0, 'groom': 0, 'ground beetle': 0, 'guacamole': 0,
        'guenon': 0, 'guillotine': 0.021521897986531258, 'guinea pig': 0, 'gyromitra': 0,
        'hair slide': 0, 'hair spray': 0.029973167926073074, 'half track': 0, 'hammer': 0,
        'hammerhead': 0, 'hamper': 0, 'hamster': 0, 'hand blower': 0.02078310213983059,
        'hand-held computer': 0, 'handkerchief': 0, 'hard disc': 0, 'hare': 0, 'harmonica': 0,
        'harp': 0, 'hartebeest': 0, 'harvester': 0, 'harvestman': 0, 'hatchet': 0, 'hay': 0,
        'head cabbage': 0, 'hen': 0, 'hen-of-the-woods': 0, 'hermit crab': 0, 'hip': 0,
        'hippopotamus': 0, 'hog': 0, 'hognose snake': 0, 'holster': 0, 'home theater': 0,
        'honeycomb': 0, 'hook': 0, 'hoopskirt': 0, 'horizontal bar': 0, 'hornbill': 0,
        'horned viper': 0, 'horse cart': 0, 'hot pot': 0, 'hotdog': 0, 'hourglass': 0,
        'house finch': 0, 'howler monkey': 0, 'hummingbird': 0, 'hyena': 0, 'iPod': 0, 'ibex': 0,
        'ice bear': 0, 'ice cream': 0, 'ice lolly': 0, 'impala': 0, 'indigo bunting': 0,
        'indri': 0, 'iron': 0, 'isopod': 0, 'jacamar': 0, "jack-o'-lantern": 0, 'jackfruit': 0,
        'jaguar': 0, 'jay': 0, 'jean': 0, 'jeep': 0, 'jellyfish': 0, 'jersey': 0,
        'jigsaw puzzle': 0, 'jinrikisha': 0, 'joystick': 0, 'junco': 0, 'keeshond': 0,
        'kelpie': 0, 'killer whale': 0, 'kimono': 0, 'king crab': 0, 'king penguin': 0,
        'king snake': 0, 'kit fox': 0, 'kite': 0, 'knee pad': 0, 'knot': 0, 'koala': 0,
        'komondor': 0, 'kuvasz': 0, 'lab coat': 0.010061823762953281, 'lacewing': 0, 'ladle': 0,
        'ladybug': 0, 'lakeside': 0, 'lampshade': 0, 'langur': 0, 'laptop': 0, 'lawn mower': 0,
        'leaf beetle': 0, 'leafhopper': 0, 'leatherback turtle': 0, 'lemon': 0, 'lens cap': 0,
        'leopard': 0, 'lesser panda': 0, 'letter opener': 0, 'library': 0, 'lifeboat': 0,
        'lighter': 0, 'limousine': 0, 'limpkin': 0, 'liner': 0, 'lion': 0, 'lionfish': 0,
        'lipstick': 0, 'little blue heron': 0, 'llama': 0, 'loggerhead': 0, 'long-horned beetle': 0,
        'lorikeet': 0, 'lotion': 0, 'loudspeaker': 0, 'loupe': 0, 'lumbermill': 0, 'lycaenid': 0,
        'lynx': 0, 'macaque': 0, 'macaw': 0, 'magnetic compass': 0, 'magpie': 0, 'mailbag': 0,
        'mailbox': 0, 'maillot': 0, 'malamute': 0, 'malinois': 0, 'manhole cover': 0, 'mantis': 0,
        'maraca': 0, 'marimba': 0, 'marmoset': 0, 'marmot': 0, 'mashed potato': 0,
        'mask': 0.01932917907834053, 'matchstick': 0, 'maypole': 0, 'maze': 0, 'measuring cup': 0,
        'meat loaf': 0, 'medicine chest': 0.013719137758016586, 'meerkat': 0, 'megalith': 0,
        'menu': 0, 'microphone': 0, 'microwave': 0, 'military uniform': 0, 'milk can': 0,
        'miniature pinscher': 0, 'miniature poodle': 0, 'miniature schnauzer': 0, 'minibus': 0,
        'miniskirt': 0, 'minivan': 0, 'mink': 0, 'missile': 0, 'mitten': 0, 'mixing bowl': 0,
        'mobile home': 0, 'modem': 0, 'monarch': 0, 'monastery': 0, 'mongoose': 0,
        'monitor': 0.013416036032140255, 'moped': 0, 'mortar': 0, 'mortarboard': 0, 'mosque': 0,
        'mosquito net': 0, 'motor scooter': 0, 'mountain bike': 0, 'mountain tent': 0, 'mouse': 0,
        'mousetrap': 0, 'moving van': 0, 'mud turtle': 0, 'mushroom': 0, 'muzzle': 0, 'nail': 0,
        'neck brace': 0, 'necklace': 0, 'nematode': 0, 'night snake': 0, 'nipple': 0,
        'notebook': 0, 'obelisk': 0, 'oboe': 0, 'ocarina': 0, 'odometer': 0, 'oil filter': 0,
        'orange': 0, 'orangutan': 0, 'organ': 0, 'oscilloscope': 0, 'ostrich': 0, 'otter': 0,
        'otterhound': 0, 'overskirt': 0, 'ox': 0, 'oxcart': 0, 'oxygen mask': 0,
        'oystercatcher': 0, 'packet': 0, 'paddle': 0, 'paddlewheel': 0, 'padlock': 0,
        'paintbrush': 0, 'pajama': 0, 'palace': 0, 'panpipe': 0, 'paper towel': 0, 'papillon': 0,
        'parachute': 0, 'parallel bars': 0, 'park bench': 0, 'parking meter': 0, 'partridge': 0,
        'passenger car': 0, 'patas': 0, 'patio': 0, 'pay-phone': 0, 'peacock': 0, 'pedestal': 0,
        'pelican': 0, 'pencil box': 0, 'pencil sharpener': 0, 'perfume': 0, 'photocopier': 0,
        'pick': 0, 'pickelhaube': 0, 'picket fence': 0, 'pickup': 0, 'pier': 0, 'piggy bank': 0,
        'pill bottle': 0, 'pillow': 0, 'pineapple': 0, 'ping-pong ball': 0, 'pinwheel': 0,
        'pirate': 0, 'pitcher': 0, 'pizza': 0, 'plane': 0, 'planetarium': 0, 'plastic bag': 0,
        'plate': 0, 'plate rack': 0, 'platypus': 0, 'plow': 0, 'plunger': 0, 'pole': 0,
        'polecat': 0, 'police van': 0, 'pomegranate': 0, 'poncho': 0, 'pool table': 0,
        'pop bottle': 0, 'porcupine': 0, 'pot': 0, 'potpie': 0, "potter's wheel": 0,
        'power drill': 0, 'prairie chicken': 0, 'prayer rug': 0, 'pretzel': 0, 'printer': 0,
        'prison': 0, 'proboscis monkey': 0, 'projectile': 0, 'projector': 0, 'promontory': 0,
        'ptarmigan': 0, 'puck': 0, 'puffer': 0, 'pug': 0, 'punching bag': 0, 'purse': 0,
        'quail': 0, 'quill': 0, 'quilt': 0, 'racer': 0, 'racket': 0, 'radiator': 0, 'radio': 0,
        'radio telescope': 0, 'rain barrel': 0, 'ram': 0, 'rapeseed': 0, 'recreational vehicle': 0,
        'red fox': 0, 'red wine': 0, 'red wolf': 0, 'red-backed sandpiper': 0,
        'red-breasted merganser': 0, 'redbone': 0, 'redshank': 0, 'reel': 0, 'reflex camera': 0,
        'refrigerator': 0, 'remote control': 0, 'restaurant': 0, 'revolver': 0,
        'rhinoceros beetle': 0, 'rifle': 0, 'ringlet': 0, 'ringneck snake': 0, 'robin': 0,
        'rock beauty': 0, 'rock crab': 0, 'rock python': 0, 'rocking chair': 0, 'rotisserie': 0,
        'rubber eraser': 0, 'ruddy turnstone': 0, 'ruffed grouse': 0, 'rugby ball': 0,
        'rule': 0.023332221433520317, 'running shoe': 0, 'safe': 0, 'safety pin': 0,
        'saltshaker': 0, 'sandal': 0, 'sandbar': 0, 'sarong': 0, 'sax': 0, 'scabbard': 0,
        'scale': 0, 'schipperke': 0, 'school bus': 0, 'schooner': 0, 'scoreboard': 0,
        'scorpion': 0, 'screen': 0.023083483800292015, 'screw': 0, 'screwdriver': 0,
        'scuba diver': 0, 'sea anemone': 0, 'sea cucumber': 0, 'sea lion': 0, 'sea slug': 0,
        'sea snake': 0, 'sea urchin': 0, 'seashore': 0, 'seat belt': 0, 'sewing machine': 0,
        'shield': 0, 'shoe shop': 0, 'shoji': 0, 'shopping basket': 0, 'shopping cart': 0,
        'shovel': 0, 'shower cap': 0, 'shower curtain': 0, 'siamang': 0, 'sidewinder': 0,
        'silky terrier': 0, 'ski': 0, 'ski mask': 0, 'skunk': 0, 'sleeping bag': 0,
        'slide rule': 0, 'sliding door': 0, 'slot': 0, 'sloth bear': 0, 'slug': 0, 'snail': 0,
        'snorkel': 0, 'snow leopard': 0, 'snowmobile': 0, 'snowplow': 0, 'soap dispenser': 0,
        'soccer ball': 0, 'sock': 0, 'soft-coated wheaten terrier': 0, 'solar dish': 0,
        'sombrero': 0, 'sorrel': 0, 'soup bowl': 0, 'space bar': 0, 'space heater': 0,
        'space shuttle': 0, 'spaghetti squash': 0, 'spatula': 0, 'speedboat': 0,
        'spider monkey': 0, 'spider web': 0, 'spindle': 0, 'spiny lobster': 0, 'spoonbill': 0,
        'sports car': 0, 'spotlight': 0.011725024320185184, 'spotted salamander': 0,
        'squirrel monkey': 0, 'stage': 0, 'standard poodle': 0, 'standard schnauzer': 0,
        'starfish': 0, 'steam locomotive': 0, 'steel arch bridge': 0, 'steel drum': 0,
        'stethoscope': 0, 'stingray': 0, 'stinkhorn': 0, 'stole': 0, 'stone wall': 0,
        'stopwatch': 0, 'stove': 0, 'strainer': 0, 'strawberry': 0, 'street sign': 0,
        'streetcar': 0, 'stretcher': 0, 'studio couch': 0, 'stupa': 0, 'sturgeon': 0,
        'submarine': 0, 'suit': 0, 'sulphur butterfly': 0, 'sulphur-crested cockatoo': 0,
        'sundial': 0, 'sunglass': 0, 'sunglasses': 0, 'sunscreen': 0, 'suspension bridge': 0,
        'swab': 0, 'sweatshirt': 0, 'swimming trunks': 0, 'swing': 0, 'switch': 0, 'syringe': 0,
        'tabby': 0, 'table lamp': 0, 'tailed frog': 0, 'tank': 0, 'tape player': 0,
        'tarantula': 0, 'teapot': 0, 'teddy': 0, 'television': 0.019651129841804504, 'tench': 0,
        'tennis ball': 0, 'terrapin': 0, 'thatch': 0, 'theater curtain': 0, 'thimble': 0,
        'three-toed sloth': 0, 'thresher': 0, 'throne': 0, 'thunder snake': 0, 'tick': 0,
        'tiger': 0, 'tiger beetle': 0, 'tiger cat': 0, 'tiger shark': 0, 'tile roof': 0,
        'timber wolf': 0, 'titi': 0, 'toaster': 0, 'tobacco shop': 0, 'toilet seat': 0,
        'toilet tissue': 0.0234963521361351, 'torch': 0, 'totem pole': 0, 'toucan': 0,
        'tow truck': 0, 'toy poodle': 0, 'toy terrier': 0, 'toyshop': 0, 'tractor': 0,
        'traffic light': 0, 'trailer truck': 0, 'tray': 0, 'tree frog': 0, 'trench coat': 0,
        'triceratops': 0, 'tricycle': 0, 'trifle': 0, 'trilobite': 0, 'trimaran': 0, 'tripod': 0,
        'triumphal arch': 0, 'trolleybus': 0, 'trombone': 0, 'tub': 0.011072015389800072,
        'turnstile': 0, 'tusker': 0, 'typewriter keyboard': 0, 'umbrella': 0, 'unicycle': 0,
        'upright': 0, 'vacuum': 0, 'valley': 0, 'vase': 0, 'vault': 0, 'velvet': 0,
        'vending machine': 0, 'vestment': 0, 'viaduct': 0, 'vine snake': 0, 'violin': 0,
        'vizsla': 0, 'volcano': 0, 'volleyball': 0, 'vulture': 0, 'waffle iron': 0,
        'walking stick': 0, 'wall clock': 0, 'wallaby': 0, 'wallet': 0, 'wardrobe': 0,
        'warplane': 0, 'warthog': 0, 'washbasin': 0.01100936345756054, 'washer': 0,
        'water bottle': 0, 'water buffalo': 0, 'water jug': 0, 'water ouzel': 0, 'water snake': 0,
        'water tower': 0, 'weasel': 0, 'web site': 0, 'weevil': 0, 'whippet': 0, 'whiptail': 0,
        'whiskey jug': 0, 'whistle': 0, 'white stork': 0, 'white wolf': 0, 'wig': 0,
        'wild boar': 0, 'window screen': 0.05903243273496628, 'window shade': 0.04712545871734619,
        'wine bottle': 0, 'wing': 0, 'wire-haired fox terrier': 0, 'wok': 0, 'wolf spider': 0,
        'wombat': 0, 'wood rabbit': 0, 'wooden spoon': 0, 'wool': 0, 'worm fence': 0, 'wreck': 0,
        'yawl': 0, "yellow lady's slipper": 0, 'yurt': 0, 'zebra': 0, 'zucchini': 0,
    }

    original_max = 0.05903243273496628  # 'window screen'

    result = normalize_ei_classification(det_classifications)

    # All output values should be parseable as floats
    softmax_values = [float(result[cls]) for cls in result]

    # Probabilities should sum to ~1.0 (small rounding error due to 4-decimal formatting)
    assert sum(softmax_values) == pytest.approx(1.0, abs=5e-3)

    # The class with the highest original confidence should have the highest normalized value
    max_class = max(result, key=lambda cls: float(result[cls]))
    assert max_class == 'window screen'

    # 'window screen' (highest input) should be strictly higher than 'window shade' (2nd highest)
    assert float(result['window screen']) > float(result['window shade'])

    # Renormalization preserves meaning: top class gets a meaningful share (~15%)
    assert float(result['window screen']) > 0.10
    # original_max value used as documentation only
    assert original_max == 0.05903243273496628

    # All values should be valid probability strings formatted to 4 decimal places
    for cls in result:
        val = result[cls]
        assert isinstance(val, str)
        assert len(val.split('.')[-1]) == 4


def test_normalize_ei_classification_with_higher_confidences():
    """Test normalization with higher confidence values where top classes are clearly distinguishable.

    Uses a classification dict where screen, monitor, computer keyboard, and desktop computer
    have significantly higher confidences, making them clearly visible after normalization.
    """
    det_classifications = {
        'Afghan hound': 0, 'African chameleon': 0, 'African crocodile': 0, 'African elephant': 0,
        'African grey': 0, 'African hunting dog': 0, 'Airedale': 0, 'American Staffordshire terrier': 0,
        'American alligator': 0, 'American black bear': 0, 'American chameleon': 0, 'American coot': 0,
        'American egret': 0, 'American lobster': 0, 'Angora': 0, 'Appenzeller': 0, 'Arabian camel': 0,
        'Arctic fox': 0, 'Australian terrier': 0, 'Band Aid': 0, 'Bedlington terrier': 0,
        'Bernese mountain dog': 0, 'Blenheim spaniel': 0, 'Border collie': 0, 'Border terrier': 0,
        'Boston bull': 0, 'Bouvier des Flandres': 0, 'Brabancon griffon': 0, 'Brittany spaniel': 0,
        'CD player': 0, 'Cardigan': 0, 'Chesapeake Bay retriever': 0, 'Chihuahua': 0,
        'Christmas stocking': 0, 'Crock Pot': 0, 'Dandie Dinmont': 0, 'Doberman': 0,
        'Dungeness crab': 0, 'Dutch oven': 0, 'Egyptian cat': 0, 'English foxhound': 0,
        'English setter': 0, 'English springer': 0, 'EntleBucher': 0, 'Eskimo dog': 0,
        'European fire salamander': 0, 'European gallinule': 0, 'French bulldog': 0, 'French horn': 0,
        'French loaf': 0, 'German shepherd': 0, 'German short-haired pointer': 0, 'Gila monster': 0,
        'Gordon setter': 0, 'Granny Smith': 0, 'Great Dane': 0, 'Great Pyrenees': 0,
        'Greater Swiss Mountain dog': 0, 'Ibizan hound': 0, 'Indian cobra': 0, 'Indian elephant': 0,
        'Irish setter': 0, 'Irish terrier': 0, 'Irish water spaniel': 0, 'Irish wolfhound': 0,
        'Italian greyhound': 0, 'Japanese spaniel': 0, 'Kerry blue terrier': 0, 'Komodo dragon': 0,
        'Labrador retriever': 0, 'Lakeland terrier': 0, 'Leonberg': 0, 'Lhasa': 0, 'Loafer': 0,
        'Madagascar cat': 0, 'Maltese dog': 0, 'Mexican hairless': 0, 'Model T': 0, 'Newfoundland': 0,
        'Norfolk terrier': 0, 'Norwegian elkhound': 0, 'Norwich terrier': 0, 'Old English sheepdog': 0,
        'Pekinese': 0, 'Pembroke': 0, 'Persian cat': 0, 'Petri dish': 0, 'Polaroid camera': 0,
        'Pomeranian': 0, 'Rhodesian ridgeback': 0, 'Rottweiler': 0, 'Saint Bernard': 0, 'Saluki': 0,
        'Samoyed': 0, 'Scotch terrier': 0, 'Scottish deerhound': 0, 'Sealyham terrier': 0,
        'Shetland sheepdog': 0, 'Shih-Tzu': 0, 'Siamese cat': 0, 'Siberian husky': 0,
        'Staffordshire bullterrier': 0, 'Sussex spaniel': 0, 'Tibetan mastiff': 0, 'Tibetan terrier': 0,
        'Walker hound': 0, 'Weimaraner': 0, 'Welsh springer spaniel': 0,
        'West Highland white terrier': 0, 'Windsor tie': 0, 'Yorkshire terrier': 0, 'abacus': 0,
        'abaya': 0, 'academic gown': 0, 'accordion': 0, 'acorn': 0, 'acorn squash': 0,
        'acoustic guitar': 0, 'admiral': 0, 'affenpinscher': 0, 'agama': 0, 'agaric': 0,
        'aircraft carrier': 0, 'airliner': 0, 'airship': 0, 'albatross': 0, 'alligator lizard': 0,
        'alp': 0, 'altar': 0, 'ambulance': 0, 'amphibian': 0, 'analog clock': 0, 'anemone fish': 0,
        'ant': 0, 'apiary': 0, 'apron': 0, 'armadillo': 0, 'artichoke': 0, 'ashcan': 0,
        'assault rifle': 0, 'axolotl': 0, 'baboon': 0, 'backpack': 0, 'badger': 0, 'bagel': 0,
        'bakery': 0, 'balance beam': 0, 'bald eagle': 0, 'balloon': 0, 'ballplayer': 0,
        'ballpoint': 0, 'banana': 0, 'banded gecko': 0, 'banjo': 0, 'bannister': 0, 'barbell': 0,
        'barber chair': 0, 'barbershop': 0, 'barn': 0, 'barn spider': 0, 'barometer': 0,
        'barracouta': 0, 'barrel': 0, 'barrow': 0, 'baseball': 0, 'basenji': 0, 'basketball': 0,
        'basset': 0, 'bassinet': 0, 'bassoon': 0, 'bath towel': 0, 'bathing cap': 0, 'bathtub': 0,
        'beach wagon': 0, 'beacon': 0, 'beagle': 0, 'beaker': 0, 'bearskin': 0, 'beaver': 0,
        'bee': 0, 'bee eater': 0, 'beer bottle': 0, 'beer glass': 0, 'bell cote': 0,
        'bell pepper': 0, 'bib': 0, 'bicycle-built-for-two': 0, 'bighorn': 0, 'bikini': 0,
        'binder': 0, 'binoculars': 0, 'birdhouse': 0, 'bison': 0, 'bittern': 0,
        'black and gold garden spider': 0, 'black grouse': 0, 'black stork': 0, 'black swan': 0,
        'black widow': 0, 'black-and-tan coonhound': 0, 'black-footed ferret': 0, 'bloodhound': 0,
        'bluetick': 0, 'boa constrictor': 0, 'boathouse': 0, 'bobsled': 0, 'bolete': 0,
        'bolo tie': 0, 'bonnet': 0, 'book jacket': 0, 'bookcase': 0, 'bookshop': 0, 'borzoi': 0,
        'bottlecap': 0, 'bow': 0, 'bow tie': 0, 'box turtle': 0, 'boxer': 0, 'brain coral': 0,
        'brambling': 0, 'brass': 0, 'brassiere': 0, 'breakwater': 0, 'breastplate': 0, 'briard': 0,
        'broccoli': 0, 'broom': 0, 'brown bear': 0, 'bubble': 0, 'bucket': 0, 'buckeye': 0,
        'buckle': 0, 'bulbul': 0, 'bull mastiff': 0, 'bullet train': 0, 'bulletproof vest': 0,
        'bullfrog': 0, 'burrito': 0, 'bustard': 0, 'butcher shop': 0, 'butternut squash': 0,
        'cab': 0, 'cabbage butterfly': 0, 'cairn': 0, 'caldron': 0, 'can opener': 0, 'candle': 0,
        'cannon': 0, 'canoe': 0, 'capuchin': 0, 'car mirror': 0, 'car wheel': 0, 'carbonara': 0,
        'cardigan': 0, 'cardoon': 0, 'carousel': 0, "carpenter's kit": 0, 'carton': 0,
        'cash machine': 0, 'cassette': 0, 'cassette player': 0, 'castle': 0, 'catamaran': 0,
        'cauliflower': 0, 'cello': 0, 'cellular telephone': 0, 'centipede': 0, 'chain': 0,
        'chain mail': 0, 'chain saw': 0, 'chainlink fence': 0, 'chambered nautilus': 0,
        'cheeseburger': 0, 'cheetah': 0, 'chest': 0, 'chickadee': 0, 'chiffonier': 0, 'chime': 0,
        'chimpanzee': 0, 'china cabinet': 0, 'chiton': 0, 'chocolate sauce': 0, 'chow': 0,
        'church': 0, 'cicada': 0, 'cinema': 0, 'cleaver': 0, 'cliff': 0, 'cliff dwelling': 0,
        'cloak': 0, 'clog': 0, 'clumber': 0, 'cock': 0, 'cocker spaniel': 0, 'cockroach': 0,
        'cocktail shaker': 0, 'coffee mug': 0, 'coffeepot': 0, 'coho': 0, 'coil': 0, 'collie': 0,
        'colobus': 0, 'combination lock': 0, 'comic book': 0, 'common iguana': 0, 'common newt': 0,
        'computer keyboard': 0.11109466105699539, 'conch': 0, 'confectionery': 0, 'consomme': 0,
        'container ship': 0, 'convertible': 0, 'coral fungus': 0, 'coral reef': 0, 'corkscrew': 0,
        'corn': 0, 'cornet': 0, 'coucal': 0, 'cougar': 0, 'cowboy boot': 0, 'cowboy hat': 0,
        'coyote': 0, 'cradle': 0, 'crane': 0, 'crash helmet': 0, 'crate': 0, 'crayfish': 0,
        'crib': 0, 'cricket': 0, 'croquet ball': 0, 'crossword puzzle': 0, 'crutch': 0,
        'cucumber': 0, 'cuirass': 0, 'cup': 0, 'curly-coated retriever': 0, 'custard apple': 0,
        'daisy': 0, 'dalmatian': 0, 'dam': 0, 'damselfly': 0, 'desk': 0.06621904671192169,
        'desktop computer': 0.0919913798570633, 'dhole': 0, 'dial telephone': 0, 'diamondback': 0,
        'diaper': 0, 'digital clock': 0, 'digital watch': 0, 'dingo': 0, 'dining table': 0,
        'dishrag': 0, 'dishwasher': 0, 'disk brake': 0, 'dock': 0, 'dogsled': 0, 'dome': 0,
        'doormat': 0, 'dough': 0, 'dowitcher': 0, 'dragonfly': 0, 'drake': 0,
        'drilling platform': 0, 'drum': 0, 'drumstick': 0, 'dugong': 0, 'dumbbell': 0,
        'dung beetle': 0, 'ear': 0, 'earthstar': 0, 'echidna': 0, 'eel': 0, 'eft': 0,
        'eggnog': 0, 'electric fan': 0, 'electric guitar': 0, 'electric locomotive': 0,
        'electric ray': 0, 'entertainment center': 0, 'envelope': 0, 'espresso': 0,
        'espresso maker': 0, 'face powder': 0, 'feather boa': 0, 'fiddler crab': 0, 'fig': 0,
        'file': 0, 'fire engine': 0, 'fire screen': 0, 'fireboat': 0, 'flagpole': 0,
        'flamingo': 0, 'flat-coated retriever': 0, 'flatworm': 0, 'flute': 0, 'fly': 0,
        'folding chair': 0, 'football helmet': 0, 'forklift': 0, 'fountain': 0, 'fountain pen': 0,
        'four-poster': 0, 'fox squirrel': 0, 'freight car': 0, 'frilled lizard': 0, 'frying pan': 0,
        'fur coat': 0, 'gar': 0, 'garbage truck': 0, 'garden spider': 0, 'garter snake': 0,
        'gas pump': 0, 'gasmask': 0, 'gazelle': 0, 'geyser': 0, 'giant panda': 0,
        'giant schnauzer': 0, 'gibbon': 0, 'go-kart': 0, 'goblet': 0, 'golden retriever': 0,
        'goldfinch': 0, 'goldfish': 0, 'golf ball': 0, 'golfcart': 0, 'gondola': 0, 'gong': 0,
        'goose': 0, 'gorilla': 0, 'gown': 0, 'grand piano': 0, 'grasshopper': 0,
        'great grey owl': 0, 'great white shark': 0, 'green lizard': 0, 'green mamba': 0,
        'green snake': 0, 'greenhouse': 0, 'grey fox': 0, 'grey whale': 0, 'grille': 0,
        'grocery store': 0, 'groenendael': 0, 'groom': 0, 'ground beetle': 0, 'guacamole': 0,
        'guenon': 0, 'guillotine': 0, 'guinea pig': 0, 'gyromitra': 0, 'hair slide': 0,
        'hair spray': 0, 'half track': 0, 'hammer': 0, 'hammerhead': 0, 'hamper': 0, 'hamster': 0,
        'hand blower': 0, 'hand-held computer': 0, 'handkerchief': 0, 'hard disc': 0, 'hare': 0,
        'harmonica': 0, 'harp': 0, 'hartebeest': 0, 'harvester': 0, 'harvestman': 0, 'hatchet': 0,
        'hay': 0, 'head cabbage': 0, 'hen': 0, 'hen-of-the-woods': 0, 'hermit crab': 0, 'hip': 0,
        'hippopotamus': 0, 'hog': 0, 'hognose snake': 0, 'holster': 0, 'home theater': 0,
        'honeycomb': 0, 'hook': 0, 'hoopskirt': 0, 'horizontal bar': 0, 'hornbill': 0,
        'horned viper': 0, 'horse cart': 0, 'hot pot': 0, 'hotdog': 0, 'hourglass': 0,
        'house finch': 0, 'howler monkey': 0, 'hummingbird': 0, 'hyena': 0, 'iPod': 0, 'ibex': 0,
        'ice bear': 0, 'ice cream': 0, 'ice lolly': 0, 'impala': 0, 'indigo bunting': 0,
        'indri': 0, 'iron': 0, 'isopod': 0, 'jacamar': 0, "jack-o'-lantern": 0, 'jackfruit': 0,
        'jaguar': 0, 'jay': 0, 'jean': 0, 'jeep': 0, 'jellyfish': 0, 'jersey': 0,
        'jigsaw puzzle': 0, 'jinrikisha': 0, 'joystick': 0, 'junco': 0, 'keeshond': 0,
        'kelpie': 0, 'killer whale': 0, 'kimono': 0, 'king crab': 0, 'king penguin': 0,
        'king snake': 0, 'kit fox': 0, 'kite': 0, 'knee pad': 0, 'knot': 0, 'koala': 0,
        'komondor': 0, 'kuvasz': 0, 'lab coat': 0, 'lacewing': 0, 'ladle': 0, 'ladybug': 0,
        'lakeside': 0, 'lampshade': 0, 'langur': 0, 'laptop': 0.024547334760427475,
        'lawn mower': 0, 'leaf beetle': 0, 'leafhopper': 0, 'leatherback turtle': 0, 'lemon': 0,
        'lens cap': 0, 'leopard': 0, 'lesser panda': 0, 'letter opener': 0, 'library': 0,
        'lifeboat': 0, 'lighter': 0, 'limousine': 0, 'limpkin': 0, 'liner': 0, 'lion': 0,
        'lionfish': 0, 'lipstick': 0, 'little blue heron': 0, 'llama': 0, 'loggerhead': 0,
        'long-horned beetle': 0, 'lorikeet': 0, 'lotion': 0, 'loudspeaker': 0, 'loupe': 0,
        'lumbermill': 0, 'lycaenid': 0, 'lynx': 0, 'macaque': 0, 'macaw': 0,
        'magnetic compass': 0, 'magpie': 0, 'mailbag': 0, 'mailbox': 0, 'maillot': 0,
        'malamute': 0, 'malinois': 0, 'manhole cover': 0, 'mantis': 0, 'maraca': 0, 'marimba': 0,
        'marmoset': 0, 'marmot': 0, 'mashed potato': 0, 'mask': 0, 'matchstick': 0, 'maypole': 0,
        'maze': 0, 'measuring cup': 0, 'meat loaf': 0, 'medicine chest': 0, 'meerkat': 0,
        'megalith': 0, 'menu': 0, 'microphone': 0, 'microwave': 0, 'military uniform': 0,
        'milk can': 0, 'miniature pinscher': 0, 'miniature poodle': 0, 'miniature schnauzer': 0,
        'minibus': 0, 'miniskirt': 0, 'minivan': 0, 'mink': 0, 'missile': 0, 'mitten': 0,
        'mixing bowl': 0, 'mobile home': 0, 'modem': 0, 'monarch': 0, 'monastery': 0,
        'mongoose': 0, 'monitor': 0.14681336283683777, 'moped': 0, 'mortar': 0, 'mortarboard': 0,
        'mosque': 0, 'mosquito net': 0, 'motor scooter': 0, 'mountain bike': 0, 'mountain tent': 0,
        'mouse': 0.012617893517017365, 'mousetrap': 0, 'moving van': 0, 'mud turtle': 0,
        'mushroom': 0, 'muzzle': 0, 'nail': 0, 'neck brace': 0, 'necklace': 0, 'nematode': 0,
        'night snake': 0, 'nipple': 0, 'notebook': 0, 'obelisk': 0, 'oboe': 0, 'ocarina': 0,
        'odometer': 0, 'oil filter': 0, 'orange': 0, 'orangutan': 0, 'organ': 0,
        'oscilloscope': 0.023647984489798546, 'ostrich': 0, 'otter': 0, 'otterhound': 0,
        'overskirt': 0, 'ox': 0, 'oxcart': 0, 'oxygen mask': 0, 'oystercatcher': 0, 'packet': 0,
        'paddle': 0, 'paddlewheel': 0, 'padlock': 0, 'paintbrush': 0, 'pajama': 0, 'palace': 0,
        'panpipe': 0, 'paper towel': 0, 'papillon': 0, 'parachute': 0, 'parallel bars': 0,
        'park bench': 0, 'parking meter': 0, 'partridge': 0, 'passenger car': 0, 'patas': 0,
        'patio': 0, 'pay-phone': 0, 'peacock': 0, 'pedestal': 0, 'pelican': 0, 'pencil box': 0,
        'pencil sharpener': 0, 'perfume': 0, 'photocopier': 0, 'pick': 0, 'pickelhaube': 0,
        'picket fence': 0, 'pickup': 0, 'pier': 0, 'piggy bank': 0, 'pill bottle': 0, 'pillow': 0,
        'pineapple': 0, 'ping-pong ball': 0, 'pinwheel': 0, 'pirate': 0, 'pitcher': 0, 'pizza': 0,
        'plane': 0, 'planetarium': 0, 'plastic bag': 0, 'plate': 0, 'plate rack': 0, 'platypus': 0,
        'plow': 0, 'plunger': 0, 'pole': 0, 'polecat': 0, 'police van': 0, 'pomegranate': 0,
        'poncho': 0, 'pool table': 0, 'pop bottle': 0, 'porcupine': 0, 'pot': 0, 'potpie': 0,
        "potter's wheel": 0, 'power drill': 0, 'prairie chicken': 0, 'prayer rug': 0, 'pretzel': 0,
        'printer': 0, 'prison': 0, 'proboscis monkey': 0, 'projectile': 0,
        'projector': 0.01448509655892849, 'promontory': 0, 'ptarmigan': 0, 'puck': 0, 'puffer': 0,
        'pug': 0, 'punching bag': 0, 'purse': 0, 'quail': 0, 'quill': 0, 'quilt': 0, 'racer': 0,
        'racket': 0, 'radiator': 0, 'radio': 0, 'radio telescope': 0, 'rain barrel': 0, 'ram': 0,
        'rapeseed': 0, 'recreational vehicle': 0, 'red fox': 0, 'red wine': 0, 'red wolf': 0,
        'red-backed sandpiper': 0, 'red-breasted merganser': 0, 'redbone': 0, 'redshank': 0,
        'reel': 0, 'reflex camera': 0, 'refrigerator': 0, 'remote control': 0, 'restaurant': 0,
        'revolver': 0, 'rhinoceros beetle': 0, 'rifle': 0, 'ringlet': 0, 'ringneck snake': 0,
        'robin': 0, 'rock beauty': 0, 'rock crab': 0, 'rock python': 0, 'rocking chair': 0,
        'rotisserie': 0, 'rubber eraser': 0, 'ruddy turnstone': 0, 'ruffed grouse': 0,
        'rugby ball': 0, 'rule': 0, 'running shoe': 0, 'safe': 0, 'safety pin': 0,
        'saltshaker': 0, 'sandal': 0, 'sandbar': 0, 'sarong': 0, 'sax': 0, 'scabbard': 0,
        'scale': 0, 'schipperke': 0, 'school bus': 0, 'schooner': 0, 'scoreboard': 0,
        'scorpion': 0, 'screen': 0.15497401356697083, 'screw': 0, 'screwdriver': 0,
        'scuba diver': 0, 'sea anemone': 0, 'sea cucumber': 0, 'sea lion': 0, 'sea slug': 0,
        'sea snake': 0, 'sea urchin': 0, 'seashore': 0, 'seat belt': 0, 'sewing machine': 0,
        'shield': 0, 'shoe shop': 0, 'shoji': 0, 'shopping basket': 0, 'shopping cart': 0,
        'shovel': 0, 'shower cap': 0, 'shower curtain': 0, 'siamang': 0, 'sidewinder': 0,
        'silky terrier': 0, 'ski': 0, 'ski mask': 0, 'skunk': 0, 'sleeping bag': 0,
        'slide rule': 0, 'sliding door': 0, 'slot': 0, 'sloth bear': 0, 'slug': 0, 'snail': 0,
        'snorkel': 0, 'snow leopard': 0, 'snowmobile': 0, 'snowplow': 0, 'soap dispenser': 0,
        'soccer ball': 0, 'sock': 0, 'soft-coated wheaten terrier': 0, 'solar dish': 0,
        'sombrero': 0, 'sorrel': 0, 'soup bowl': 0, 'space bar': 0, 'space heater': 0,
        'space shuttle': 0, 'spaghetti squash': 0, 'spatula': 0, 'speedboat': 0,
        'spider monkey': 0, 'spider web': 0, 'spindle': 0, 'spiny lobster': 0, 'spoonbill': 0,
        'sports car': 0, 'spotlight': 0, 'spotted salamander': 0, 'squirrel monkey': 0, 'stage': 0,
        'standard poodle': 0, 'standard schnauzer': 0, 'starfish': 0, 'steam locomotive': 0,
        'steel arch bridge': 0, 'steel drum': 0, 'stethoscope': 0, 'stingray': 0, 'stinkhorn': 0,
        'stole': 0, 'stone wall': 0, 'stopwatch': 0, 'stove': 0, 'strainer': 0, 'strawberry': 0,
        'street sign': 0, 'streetcar': 0, 'stretcher': 0, 'studio couch': 0, 'stupa': 0,
        'sturgeon': 0, 'submarine': 0, 'suit': 0, 'sulphur butterfly': 0,
        'sulphur-crested cockatoo': 0, 'sundial': 0, 'sunglass': 0, 'sunglasses': 0,
        'sunscreen': 0, 'suspension bridge': 0, 'swab': 0, 'sweatshirt': 0, 'swimming trunks': 0,
        'swing': 0, 'switch': 0, 'syringe': 0, 'tabby': 0, 'table lamp': 0, 'tailed frog': 0,
        'tank': 0, 'tape player': 0, 'tarantula': 0, 'teapot': 0, 'teddy': 0, 'television': 0,
        'tench': 0, 'tennis ball': 0, 'terrapin': 0, 'thatch': 0, 'theater curtain': 0,
        'thimble': 0, 'three-toed sloth': 0, 'thresher': 0, 'throne': 0, 'thunder snake': 0,
        'tick': 0, 'tiger': 0, 'tiger beetle': 0, 'tiger cat': 0, 'tiger shark': 0, 'tile roof': 0,
        'timber wolf': 0, 'titi': 0, 'toaster': 0, 'tobacco shop': 0, 'toilet seat': 0,
        'toilet tissue': 0, 'torch': 0, 'totem pole': 0, 'toucan': 0, 'tow truck': 0,
        'toy poodle': 0, 'toy terrier': 0, 'toyshop': 0, 'tractor': 0, 'traffic light': 0,
        'trailer truck': 0, 'tray': 0, 'tree frog': 0, 'trench coat': 0, 'triceratops': 0,
        'tricycle': 0, 'trifle': 0, 'trilobite': 0, 'trimaran': 0, 'tripod': 0,
        'triumphal arch': 0, 'trolleybus': 0, 'trombone': 0, 'tub': 0, 'turnstile': 0, 'tusker': 0,
        'typewriter keyboard': 0, 'umbrella': 0, 'unicycle': 0, 'upright': 0, 'vacuum': 0,
        'valley': 0, 'vase': 0, 'vault': 0, 'velvet': 0, 'vending machine': 0, 'vestment': 0,
        'viaduct': 0, 'vine snake': 0, 'violin': 0, 'vizsla': 0, 'volcano': 0, 'volleyball': 0,
        'vulture': 0, 'waffle iron': 0, 'walking stick': 0, 'wall clock': 0, 'wallaby': 0,
        'wallet': 0, 'wardrobe': 0, 'warplane': 0, 'warthog': 0, 'washbasin': 0, 'washer': 0,
        'water bottle': 0, 'water buffalo': 0, 'water jug': 0, 'water ouzel': 0, 'water snake': 0,
        'water tower': 0, 'weasel': 0, 'web site': 0.013706184923648834, 'weevil': 0, 'whippet': 0,
        'whiptail': 0, 'whiskey jug': 0, 'whistle': 0, 'white stork': 0, 'white wolf': 0,
        'wig': 0, 'wild boar': 0, 'window screen': 0, 'window shade': 0.017799241468310356,
        'wine bottle': 0, 'wing': 0, 'wire-haired fox terrier': 0, 'wok': 0, 'wolf spider': 0,
        'wombat': 0, 'wood rabbit': 0, 'wooden spoon': 0, 'wool': 0, 'worm fence': 0, 'wreck': 0,
        'yawl': 0, "yellow lady's slipper": 0, 'yurt': 0, 'zebra': 0, 'zucchini': 0,
    }

    result = normalize_ei_classification(det_classifications)

    # All output values should be parseable as floats
    softmax_values = [float(result[cls]) for cls in result]

    # Probabilities should sum to ~1.0
    assert sum(softmax_values) == pytest.approx(1.0, abs=5e-3)

    # 'screen' and 'monitor' have the highest confidences (0.155, 0.147)
    # They should both be at the top tier (above zero-confidence classes)
    screen_val = float(result['screen'])
    monitor_val = float(result['monitor'])
    keyboard_val = float(result['computer keyboard'])
    desktop_val = float(result['desktop computer'])
    desk_val = float(result['desk'])
    zero_val = float(result['Afghan hound'])

    # Top classes should be distinguishable from zero-confidence classes
    assert screen_val > zero_val
    assert monitor_val > zero_val
    assert keyboard_val > zero_val
    assert desktop_val > zero_val
    assert desk_val > zero_val

    # Zero-input classes get exactly 0.0
    assert zero_val == 0.0

    # Strict ordering matches the raw input ranking (sum of non-zero ~0.67)
    assert screen_val > monitor_val
    assert monitor_val > keyboard_val
    assert keyboard_val > desktop_val
    assert desktop_val > desk_val

    # Top class should get a meaningful percentage (>20%)
    assert screen_val > 0.20


def test_renormalization_over_non_zero_produces_meaningful_percentages():
    """Test that simple renormalization (divide by sum of non-zero) produces meaningful percentages.

    Uses a real EI classification snapshot of a monitor displaying a webcam preview.
    The raw EI values already represent probability-like scores and renormalizing them
    over the non-zero subset preserves their relative magnitudes — unlike softmax which
    collapses everything toward uniform when applied over 1000 classes.

    The expected top classes (monitor, web site, screen, desktop computer) should each
    receive >10% probability, in stark contrast to the ~0.12% softmax produces.
    """
    # Real EI capture of a monitor showing a webcam preview (only non-zero values shown;
    # the function under test handles dicts with ~1000 classes mostly at 0.0)
    raw = {
        'binder': 0.018793530762195587,
        'cash machine': 0.014952240511775017,
        'desk': 0.019223056733608246,
        'desktop computer': 0.0947352722287178,
        'envelope': 0.014027869328856468,
        'iPod': 0.010632287710905075,
        'laptop': 0.035293109714984894,
        'monitor': 0.20643608272075653,
        'mouse': 0.032460279762744904,
        'notebook': 0.03125991299748421,
        'screen': 0.1367611289024353,
        'web site': 0.13983964920043945,
        'Afghan hound': 0,
    }
    # Pad to ~1000 classes with zero-confidence to mirror real EI output
    raw.update({f'_zero_{i}': 0 for i in range(987)})

    result = normalize_ei_classification(raw)

    # All probabilities (as floats) should sum to ~1.0 (small rounding from 4-decimal formatting)
    assert sum(float(v) for v in result.values()) == pytest.approx(1.0, abs=5e-3)

    # The top class should be 'monitor' (it has the highest raw value)
    max_class = max(result, key=lambda k: float(result[k]))
    assert max_class == 'monitor'

    # Top classes should receive meaningful percentages (>10%), not collapsed to ~0.1%
    assert float(result['monitor']) > 0.20  # ~27%
    assert float(result['web site']) > 0.15  # ~19%
    assert float(result['screen']) > 0.15  # ~18%
    assert float(result['desktop computer']) > 0.10  # ~13%

    # Strict ordering matches the raw input ordering
    assert float(result['monitor']) > float(result['web site'])
    assert float(result['web site']) > float(result['screen'])
    assert float(result['screen']) > float(result['desktop computer'])
    assert float(result['desktop computer']) > float(result['laptop'])

    # Zero-input classes stay at exactly 0
    assert result['Afghan hound'] == "0.0000"
    assert result['_zero_0'] == "0.0000"

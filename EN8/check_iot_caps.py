"""Check jsonld-ex IoT pipeline capabilities."""
import sys
sys.path.insert(0, "packages/python/src")
import jsonld_ex as jx
import json

doc = {
    '@context': {'@vocab': 'https://schema.org/'},
    '@type': 'Observation',
    '@id': 'obs-001',
    'temperature': jx.annotate(value=22.5, confidence=0.95, source='sensor-4',
        measurement_uncertainty=0.25, unit='celsius',
        calibrated_at='2024-01-15T10:00:00Z'),
}

# CBOR-LD
cbor_bytes = jx.to_cbor(doc)
json_bytes = json.dumps(doc).encode()
print(f"=== CBOR-LD ===")
print(f"JSON size: {len(json_bytes)} bytes")
print(f"CBOR size: {len(cbor_bytes)} bytes")
print(f"Ratio: {len(cbor_bytes)/len(json_bytes):.1%}")

# MQTT
mqtt = jx.to_mqtt_payload(doc)
print(f"\n=== MQTT ===")
print(f"Topic: {jx.derive_mqtt_topic(doc)}")
print(f"QoS: {jx.derive_mqtt_qos(doc)}")
print(f"Payload size: {len(mqtt)} bytes")

# CoAP
coap = jx.to_coap_payload(doc)
print(f"\n=== CoAP ===")
print(f"Payload size: {len(coap)} bytes")
coap_opts = jx.derive_coap_options(doc)
print(f"Options: {coap_opts}")

# validate_document signature
print(f"\n=== validate_document signature ===")
help(jx.validate_document)

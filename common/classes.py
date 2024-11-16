from enum import Enum


class DetectionClassesName(Enum):
    SmallVehicle = "SmallVehicle",  # Alfred
    BigVehicle = "BigVehicle",      # Alfred

    LightVehicle = "LightVehicle",  # Waldo
    Person = "Person",              # Waldo     #
    Building = "Building",          # Waldo
    UPole = "UPole",                # Waldo
    Boat = "Boat",                  # Waldo
    Bike = "Bike",                  # Waldo
    Container = "Container",        # Waldo
    Truck = "Truck",                # Waldo
    Gastank = "Gastank",            # Waldo
    Digger = "Digger",              # Waldo
    Solarpanels = "Solarpanels",    # Waldo
    Bus = "Bus",                    # Waldo


class SegmentationClassesName(Enum):
    unknown = "unknown",            # Open Earth Map
    greenery = "greenery",          # Open Earth Map
    pavement = "pavement",          # Open Earth Map
    road = "road",                  # Open Earth Map
    buildings = "buildings",        # Open Earth Map
    water = "water",                # Open Earth Map

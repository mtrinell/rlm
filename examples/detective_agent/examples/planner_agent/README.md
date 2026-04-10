# Travel Planner

End-to-end travel planning application with multiple specialized agents working together to search and book flights, hotels, restaurants, and transfers, then produce a complete, costed itinerary.

## App Overview

- **Framework**: LangGraph application
- **LLM calls**: LiteLLM SDK
- **Tools**: MCP tools
- **Invocation**: Direct `LangGraph workflow.invoke()` or A2A server (A2A SDK)
- **Observability**: IOA Observe SDK for traces

---

## Agents

### Orchestrator

First node. Parses the free-text user request and extracts structured travel details (origin, destination, departure/return dates, traveler count, budget, preferences). Sets `current_step = "flight_search"` and populates the shared graph state for downstream agents.

**Tools**: —

---

### Flight Agent

Searches for available flights matching origin, destination, date, passenger count, and class. Picks the best option balancing price and schedule, then books it.

**Tools**: `search_flights`, `book_flight`, `get_flight_details`, `cancel_flight`, `get_all_flight_bookings`

---

### Hotel Agent

Searches for hotels at the destination for the stay dates. Applies a budget-derived `max_price` filter when a budget was provided. Passes the raw `user_request` verbatim into its LLM prompt as "Guest preferences / original request". Picks the best value option and books it.

**Tools**: `search_hotels`, `book_hotel`, `get_hotel_details`, `cancel_hotel`, `get_all_hotel_bookings`

---

### Restaurant Agent

Searches for a dinner reservation at the destination on the departure date at 19:00. Filters by cuisine preference when present. Books one restaurant.

**Tools**: `search_restaurants`, `book_restaurant`, `get_reservation_details`, `cancel_reservation`, `get_all_reservations`

---

### Transfer Agent

Searches for airport-transfer options at the destination. Optionally also searches for car-rental options. Books one transfer. Last booking node in the pipeline.

**Tools**: `search_transfers`, `book_transfer`, `get_transfer_details`, `cancel_transfer`, `get_all_transfer_bookings`

---

### Summarizer

Last node. Collects all confirmed bookings from state, computes the total trip cost, and renders a structured Markdown itinerary (flights, hotel, restaurant, transfers). Produces the final user-facing output.

**Tools**: —

---

## Tools

### Flight Tools

| Tool | Description | Parameters | Return |
|------|-------------|------------|--------|
| `search_flights` | Full-text search over the flight catalogue. Filters by origin city, destination city, departure date prefix, seat class, and minimum available seats. | `from_location` (str), `to_location` (str), `departure_date` (str, YYYY-MM-DD), `passengers` (int, default 1), `flight_class` (str, default "economy") | `list[dict]` |
| `book_flight` | Creates a confirmed flight booking, decrements available seats, persists to `flights.json`, and returns full booking details including a computed check-in time 3 hours before departure. | `flight_id` (str, required), `passenger_name` (str, required), `passengers` (int, default 1), `email` (str) | `dict` |
| `get_flight_details` | Returns full details for a single flight by ID. | `flight_id` (str, required) | `dict \| None` |
| `cancel_flight` | Cancels a confirmed flight booking by booking ID, restores the seat count on the flight record, and marks the booking status as `"cancelled"`. | `booking_id` (str, required) | `dict` |
| `get_all_flight_bookings` | Returns all flight booking records (all statuses). | — | `list[dict]` |

### Hotel Tools

| Tool | Description | Parameters | Return |
|------|-------------|------------|--------|
| `search_hotels` | Searches hotels by exact location match, filters out fully-booked properties, enforces a `min_rating` floor, and optionally caps on `max_price` per night. | `location` (str, required), `check_in` (str, required), `check_out` (str, required), `guests` (int, default 1), `min_rating` (float, default 0.0), `max_price` (float \| None) | `list[dict]` |
| `book_hotel` | Books one room, calculates total price from nights × price-per-night, decrements available rooms, persists to `hotels.json`. | `hotel_id` (str, required), `guest_name` (str, required), `check_in` (str, required), `check_out` (str, required), `guests` (int, default 1), `email` (str) | `dict` |
| `get_hotel_details` | Returns full details for a single hotel by ID. | `hotel_id` (str, required) | `dict \| None` |
| `cancel_hotel` | Cancels a confirmed hotel booking by booking ID, restores one available room on the hotel record, and marks the booking status as `"cancelled"`. | `booking_id` (str, required) | `dict` |
| `get_all_hotel_bookings` | Returns all hotel booking records (all statuses). | — | `list[dict]` |

### Restaurant Tools

| Tool | Description | Parameters | Return |
|------|-------------|------------|--------|
| `search_restaurants` | Searches restaurants by exact location match, optional exact-string cuisine filter, optional available time-slot filter, and `min_rating` floor. | `location` (str, required), `cuisine` (str \| None), `date` (str \| None), `time` (str \| None), `min_rating` (float, default 0.0) | `list[dict]` |
| `book_restaurant` | Creates a confirmed reservation, removes the booked time slot from `available_slots`, persists to `restaurants.json`. | `restaurant_id` (str, required), `guest_name` (str, required), `date` (str, required), `time` (str, required), `party_size` (int, default 2), `email` (str) | `dict` |
| `get_reservation_details` | Returns the full details of a single restaurant reservation by reservation ID. | `reservation_id` (str, required) | `dict \| None` |
| `cancel_reservation` | Cancels a restaurant reservation, restores the time slot to the restaurant's `available_slots`, and marks the reservation status as `"cancelled"`. | `reservation_id` (str, required) | `dict` |
| `get_all_reservations` | Returns all restaurant reservation records (all statuses). | — | `list[dict]` |

### Transfer Tools

| Tool | Description | Parameters | Return |
|------|-------------|------------|--------|
| `search_transfers` | Searches transfers by location (or by `from_location` for point-to-point transfers), optional `transfer_type` filter (`airport_transfer`, `car_rental`), and availability check. | `location` (str, required), `transfer_type` (str \| None), `from_location` (str \| None), `to_location` (str \| None) | `list[dict]` |
| `book_transfer` | Books a transfer or car rental, computes total price (`price_per_day × days` for rentals, flat price for transfers), decrements integer availability counters, persists to `transfers.json`. | `transfer_id` (str, required), `customer_name` (str, required), `pickup_date` (str, required), `pickup_time` (str \| None), `days` (int, default 1), `email` (str) | `dict` |
| `get_transfer_details` | Returns full details for a single transfer or car-rental option by ID. | `transfer_id` (str, required) | `dict \| None` |
| `cancel_transfer` | Cancels a transfer booking by booking ID, increments the integer availability counter on the transfer record (if applicable), and marks the booking status as `"cancelled"`. | `booking_id` (str, required) | `dict` |
| `get_all_transfer_bookings` | Returns all transfer booking records (all statuses). | — | `list[dict]` |

---

## Use Case Examples

### "Book a week in Paris from London, 2 travelers, budget €4000"

| Agent | Actions |
|-------|---------|
| **Orchestrator** | Parses request → `origin="London"`, `destination="Paris"`, `departure_date="<date>"`, `return_date="<date+7>"`, `travelers=2`, `budget=4000.0` |
| **Flight Agent** | `search_flights(from_location="London", to_location="Paris", departure_date="<date>", passengers=2, flight_class="economy")` → `book_flight(flight_id="FL...", passenger_name="Traveler", passengers=2)` |
| **Hotel Agent** | `max_price` derived from budget: `4000 × 0.3 / 7 ≈ 171.43` → `search_hotels(location="Paris", check_in="<date>", check_out="<date+7>", guests=2, max_price=171.43)` → `book_hotel(hotel_id="HT...", guest_name="Traveler", check_in="<date>", check_out="<date+7>", guests=2)` |
| **Restaurant Agent** | `search_restaurants(location="Paris", date="<date>", time="19:00")` → `book_restaurant(restaurant_id="RS...", guest_name="Traveler", date="<date>", time="19:00", party_size=2)` |
| **Transfer Agent** | `search_transfers(location="Paris", transfer_type="airport_transfer")` → `book_transfer(transfer_id="TR...", customer_name="Traveler", pickup_date="<date>", pickup_time="10:00")` |
| **Summarizer** | Aggregates all bookings, computes total cost, renders final Markdown itinerary. |

---

### "Weekend trip to London, depart Friday, 1 traveler, Italian food preferred"

| Agent | Actions |
|-------|---------|
| **Orchestrator** | Parses → `destination="London"`, `preferences={"cuisine": "Italian"}`, `travelers=1` |
| **Flight Agent** | `search_flights(from_location="<inferred origin>", to_location="London", departure_date="<Friday>", passengers=1)` → `book_flight(...)` |
| **Hotel Agent** | No budget constraint → `max_price=None` → `search_hotels(location="London", check_in="<Friday>", check_out="<Sunday>", guests=1)` → `book_hotel(...)` |
| **Restaurant Agent** | `search_restaurants(location="London", cuisine="Italian", date="<Friday>", time="19:00")` → `book_restaurant(restaurant_id="RS002", guest_name="Traveler", date="<Friday>", time="19:00", party_size=1)` ← Pasta Paradise selected |
| **Transfer Agent** | `search_transfers(location="London", transfer_type="airport_transfer")` → `book_transfer(...)` |
| **Summarizer** | Renders final itinerary. |

# Re-slotting report

Slotting is fitted on weeks 1-8 (20163 orders) and scored on those weeks and on held-out weeks 9-12 (10105 orders). Travel is S-shape routing with carts of 8 orders in arrival order; the last column re-scores the held-out weeks under largest-gap routing.

## Replay

| plan | SKUs moved | training m/order | held-out m/order | training vs current | held-out vs current | held-out m/order, largest-gap |
|---|---:|---:|---:|---:|---:|---:|
| current | 0 | 32.56 | 32.85 | - | - | 27.28 |
| full re-slot, depot | 1500 | 15.80 | 16.94 | -51.5% | -48.4% | 14.34 |
| full re-slot, aisle | 1496 | 13.69 | 14.90 | -58.0% | -54.6% | 13.14 |
| aisle, 50 moves | 50 | 28.86 | 29.62 | -11.4% | -9.8% | 24.17 |
| aisle, 150 moves | 147 | 23.94 | 24.86 | -26.5% | -24.3% | 20.76 |
| aisle, 300 moves | 291 | 20.75 | 21.71 | -36.3% | -33.9% | 18.10 |
| aisle, unlimited moves | 1287 | 16.47 | 17.36 | -49.4% | -47.2% | 15.20 |

`full re-slot, aisle` saves 58.0% of travel per order on the weeks it was fitted on (weeks 1-8) but 54.6% on the held-out weeks (9-12): an optimism gap of 3.3 points.

## Top moves

The 10 moves with the largest proxy saving in the 150-move aisle plan (150 moves in all).

| rank | sku | product | from | to | proxy saving (pick-metres/week) |
|---:|---|---|---|---|---:|
| 1 | SKU1156 | Lumen Bamboo Mascara | A07-R07-3 | A01-L01-3 | 17681.2 |
| 2 | SKU0592 | Sorrel Mint Night Cream | A07-L22-4 | A01-R01-3 | 10146.4 |
| 3 | SKU1371 | Lumen Tea Tree Mascara | A06-R22-2 | A01-L01-2 | 5489.1 |
| 4 | SKU0633 | Oakmoor Bamboo Face Serum | A06-L01-4 | A01-R14-4 | 2300.1 |
| 5 | SKU1330 | Sorrel Charcoal Magnesium Tablets | A07-R09-4 | A01-L03-3 | 1990.1 |
| 6 | SKU0631 | Oakmoor Citrus Toothpaste | A09-L05-1 | A01-R04-3 | 1629.6 |
| 7 | SKU0318 | Oakmoor Tea Tree Shave Foam | A09-L02-1 | A01-L04-2 | 1565.1 |
| 8 | SKU0658 | Sorrel Marine Razor Cartridges | A06-R05-4 | A01-R03-2 | 1494.6 |
| 9 | SKU0981 | Novara Ginger Lip Tint | A04-R08-4 | A01-L02-3 | 1482.3 |
| 10 | SKU1246 | Novara Oat Vitamin C | A10-R20-4 | A01-R05-2 | 1384.5 |

## Affinity pairs

25 pairs with support >= 0.5% and lift >= 3 in the training weeks. The affinity pass accepted 0 swaps for the aisle objective and 8 for the depot objective.

| sku a | sku b | support | lift | same aisle in aisle plan |
|---|---|---:|---:|---|
| SKU0010 Sorrel Citrus Mouthwash | SKU0629 Sorrel Citrus Toothpaste | 0.95% | 101.3 | yes |
| SKU0960 Aurel Bamboo Body Lotion | SKU1445 Aurel Bamboo Body Wash | 0.95% | 91.2 | yes |
| SKU0819 Halden Marine Shave Gel | SKU1485 Halden Marine Aftershave Balm | 0.94% | 101.8 | yes |
| SKU0512 Lumen Marine Moisturiser | SKU0600 Lumen Marine Face Wash | 0.94% | 96.3 | yes |
| SKU0289 Oakmoor Peppermint Body Wash | SKU1043 Oakmoor Peppermint Body Lotion | 0.94% | 70.4 | yes |
| SKU0210 Fernhill Bamboo Toothpaste | SKU0255 Fernhill Bamboo Mouthwash | 0.93% | 100.2 | yes |
| SKU1212 Calder Tea Tree Shampoo | SKU1317 Calder Tea Tree Conditioner | 0.92% | 93.8 | yes |
| SKU0835 Veda Argan Face Wash | SKU1184 Veda Argan Moisturiser | 0.89% | 103.2 | yes |
| SKU0220 Oakmoor Ginger Toner | SKU1397 Oakmoor Ginger Cleanser | 0.87% | 108.3 | yes |
| SKU0232 Lumen Peppermint Night Serum | SKU0639 Lumen Peppermint Eye Cream | 0.85% | 69.8 | yes |
| SKU1122 Oakmoor Charcoal Lip Scrub | SKU1195 Oakmoor Charcoal Lip Balm | 0.84% | 104.8 | yes |
| SKU0338 Lumen Mint Cotton Pads | SKU1420 Lumen Mint Micellar Water | 0.80% | 109.3 | yes |
| SKU1006 Novara Citrus Cleanser | SKU1279 Novara Citrus Toner | 0.78% | 117.7 | yes |
| SKU0170 Calder Oat Aftershave Balm | SKU0884 Calder Oat Shave Gel | 0.77% | 119.9 | yes |
| SKU0468 Aurel Rose Body Spray | SKU0789 Aurel Rose Deodorant | 0.76% | 122.1 | yes |
| SKU0042 Fernhill Aloe Dry Shampoo | SKU0939 Fernhill Aloe Hair Spray | 0.74% | 127.5 | yes |
| SKU0223 Novara Oat Hair Mask | SKU1343 Novara Oat Leave-in Serum | 0.72% | 115.8 | yes |
| SKU0008 Novara Shea Lip Balm | SKU0105 Novara Shea Lip Scrub | 0.72% | 127.6 | yes |
| SKU0829 Veda Cedar Hair Mask | SKU1318 Veda Cedar Leave-in Serum | 0.72% | 83.1 | yes |
| SKU0028 Aurel Argan Shampoo | SKU1424 Aurel Argan Conditioner | 0.71% | 118.7 | yes |
| SKU0268 Veda Rose Cotton Pads | SKU0569 Veda Rose Micellar Water | 0.71% | 114.1 | yes |
| SKU0919 Halden Charcoal Hand Wash | SKU1198 Halden Charcoal Hand Cream | 0.66% | 136.9 | yes |
| SKU0845 Fernhill Cedar Hand Cream | SKU1264 Fernhill Cedar Hand Wash | 0.64% | 144.9 | yes |
| SKU0901 Sorrel Lavender Eye Cream | SKU1136 Sorrel Lavender Night Serum | 0.59% | 106.0 | yes |
| SKU0614 Calder Lavender Body Spray | SKU1441 Calder Lavender Deodorant | 0.55% | 147.2 | yes |

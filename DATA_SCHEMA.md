# NFL Big Data Bowl 2026 - Data Schema

## Competition Overview
- **Task**: Predict player movement (x, y coordinates) while the ball is in the air
- **Metric**: Custom metric (RMSE-based continuous coordinate prediction)
- **Training data**: 18 weeks of 2023 NFL season (weekly input/output CSV pairs)
- **Test data**: 5,837 unlabeled frames requiring predictions

## Data Files

### Training Input Files
**Pattern**: `train/input_2023_w{01..18}.csv`
**Size**: ~4.88M rows total across all weeks

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| game_id | str | Unique game identifier |
| play_id | int | Play identifier within game |
| player_to_predict | int | NFL player ID to predict |
| nfl_id | int | NFL player ID |
| frame_id | int | Frame number in play |
| play_direction | float | Direction of play (0-360 degrees) |
| absolute_yardline_number | int | Yardline position (0-120) |
| player_name | str | Player name |
| player_height | float | Player height |
| player_weight | float | Player weight |
| player_birth_date | str | Player birth date |
| player_position | str | Player position (WR, TE, LB, etc.) |
| player_side | str | Player side (offense/defense) |
| player_role | str | Player role in play |
| x | float | Current x-coordinate (field position) |
| y | float | Current y-coordinate (field position) |
| s | float | Speed (yards/second) |
| a | float | Acceleration |
| dir | float | Direction of movement (radians) |
| o | float | Orientation angle |
| num_frames_output | int | Number of frames to predict ahead |
| ball_land_x | float | Where ball lands (x) |
| ball_land_y | float | Where ball lands (y) |

### Training Output Files
**Pattern**: `train/output_2023_w{01..18}.csv`
**Size**: ~563k rows (subset of input; only frames with ground truth labels)

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| game_id | str | Game identifier |
| play_id | int | Play identifier |
| nfl_id | int | NFL player ID |
| frame_id | int | Frame number |
| x | float | Target x-coordinate (future position) |
| y | float | Target y-coordinate (future position) |

### Test Input File
**File**: `test_input.csv`
**Size**: 49,753 rows

Same schema as training input files (includes all features needed for prediction).

### Test File (Unlabeled)
**File**: `test.csv`
**Size**: 5,837 rows

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| id | int | Unique row identifier for submission |
| game_id | str | Game identifier |
| play_id | int | Play identifier |
| nfl_id | int | NFL player ID |
| frame_id | int | Frame number |

## Submission Format

**File**: CSV with exactly 2 columns

| Column | Type | Description |
|--------|------|-------------|
| x | float | Predicted x-coordinate |
| y | float | Predicted y-coordinate |

**Requirements**:
- Exactly 5,837 rows (matching `test.csv` row count)
- No NaN values
- All values must be finite (no Inf/-Inf)
- No additional columns (especially not `id`, `game_id`, etc.)
- Column order must be: `x`, `y`

**Example** (first 3 rows):
```
x,y
84.405595,41.289776
84.192087,41.438702
85.568416,41.963424
```

## Key Insights

### Data Characteristics
- **Multi-temporal**: Each row includes past frame data and must predict future player positions
- **Per-play batching**: Predictions grouped by `game_id` + `play_id` for validation
- **Position normalization**: x and y are in field coordinates (0-120 for length, 0-~54 for width)
- **Rich features**: Player biometics, orientation, speed, acceleration already computed
- **Ball landing info**: Included to help predict player trajectory (intelligent relocation toward ball landing spot)

### Training / Validation Split Considerations
- **Natural play-level split**: To avoid leakage, split by complete plays (all frames in a play go to same split)
- **Temporal ordering**: 18 weeks of data; earlier weeks for train, later for validation (if desired)
- **Class imbalance**: Different numbers of frames per play; weighted evaluation may apply

### Feature Engineering Opportunities
- **Ball-relative features**: Distance/direction to ball landing spot
- **Time-lag features**: Velocity/acceleration trends over past frames
- **Play context**: Offensive/defensive role, position-specific patterns
- **Normalized coordinates**: Field-relative vs. absolute positioning

## Model Baseline Notes

Current implementation:
- **Train data**: ~560k merged rows (after joining input/output on keys)
- **Features used**: 12 engineered features including ball-relative and time-lag
- **Model**: HistGradientBoostingRegressor (x and y regressors)
- **Validation RMSE**: ~3.47 (combined on sampled holdout)

**Potential improvements**:
- Train on full dataset (not sampled)
- Play-level cross-validation to avoid temporal leakage
- Ensemble methods or LightGBM/XGBoost
- More sophisticated feature engineering
- Hyperparameter tuning per position/role

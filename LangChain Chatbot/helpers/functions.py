from typing import List, Dict, Any, Set
from helpers.security import sanitize_workout_input
from dataclasses import dataclass

class WorkoutGenerationError(Exception):
    """Custom exception for workout generation errors"""
    pass

def validate_workout_inputs(muscle_group: str, experience_level: str, equipment_available: List[str]) -> None:
    """Validate workout generation inputs"""
    valid_muscle_groups = {'chest', 'back', 'legs', 'shoulders', 'arms', 'full body', 'core'}
    valid_experience_levels = {'beginner', 'intermediate', 'advanced'}
    valid_equipment = {'dumbbells', 'barbell', 'cable machine', 'bodyweight only', 'full gym'}

    if not muscle_group or muscle_group.lower() not in valid_muscle_groups:
        raise WorkoutGenerationError(f"Invalid muscle group. Must be one of: {', '.join(valid_muscle_groups)}")
    
    if not experience_level or experience_level.lower() not in valid_experience_levels:
        raise WorkoutGenerationError(f"Invalid experience level. Must be one of: {', '.join(valid_experience_levels)}")
    
    if not equipment_available:
        raise WorkoutGenerationError("Must specify at least one piece of equipment")
    
    if not all(eq.lower() in valid_equipment for eq in equipment_available):
        raise WorkoutGenerationError(f"Invalid equipment. Must be from: {', '.join(valid_equipment)}")

def generate_workout(muscle_group: str, experience_level: str, equipment_available: List[str]) -> List[str]:
    """
    Generate a workout plan based on muscle group, experience level, and available equipment.
    
    Args:
        muscle_group (str): Target muscle group
        experience_level (str): User's experience level
        equipment_available (List[str]): List of available equipment
    
    Returns:
        List[str]: List of exercises with sets and reps
    
    Raises:
        WorkoutGenerationError: If inputs are invalid or no suitable exercises can be found
    """
    # validate inputs
    validate_workout_inputs(muscle_group, experience_level, equipment_available)
    
    # convert inputs to lowercase for consistency
    muscle_group = muscle_group.lower()
    experience_level = experience_level.lower()
    equipment_available = [eq.lower() for eq in equipment_available]

    # comprehensive exercise database
    workouts = {
        'chest': {
            'beginner': [
                {'exercise': 'Push-ups (3x12)', 'equipment': ['bodyweight only']},
                {'exercise': 'Machine Chest Press (3x10)', 'equipment': ['full gym']},
                {'exercise': 'Dumbbell Bench Press (3x10)', 'equipment': ['dumbbells']},
                {'exercise': 'Barbell Bench Press (3x8)', 'equipment': ['barbell']},
                {'exercise': 'Incline Dumbbell Press (3x10)', 'equipment': ['dumbbells']},
                {'exercise': 'Cable Chest Flyes (3x12)', 'equipment': ['cable machine', 'full gym']}
            ],
            'intermediate': [
                {'exercise': 'Barbell Bench Press (4x8)', 'equipment': ['barbell']},
                {'exercise': 'Incline Dumbbell Press (4x10)', 'equipment': ['dumbbells']},
                {'exercise': 'Dumbbell Flyes (3x12)', 'equipment': ['dumbbells']},
                {'exercise': 'Cable Flyes (3x12)', 'equipment': ['cable machine']},
                {'exercise': 'Push-ups with Elevation (4x12)', 'equipment': ['bodyweight only']},
                {'exercise': 'Machine Pec Deck (3x12)', 'equipment': ['full gym']}
            ],
            'advanced': [
                {'exercise': 'Incline Barbell Press (4x8)', 'equipment': ['barbell']},
                {'exercise': 'Decline Barbell Press (4x8)', 'equipment': ['barbell']},
                {'exercise': 'Cable Crossovers (4x12)', 'equipment': ['cable machine']},
                {'exercise': 'Dumbbell Press (4x8)', 'equipment': ['dumbbells']},
                {'exercise': 'Weighted Dips (4x10)', 'equipment': ['bodyweight only', 'full gym']},
                {'exercise': 'Paused Bench Press (5x5)', 'equipment': ['barbell']}
            ]
        },
        'back': {
            'beginner': [
                {'exercise': 'Assisted Pull-ups (3x8)', 'equipment': ['full gym']},
                {'exercise': 'Lat Pulldowns (3x10)', 'equipment': ['cable machine', 'full gym']},
                {'exercise': 'Seated Cable Rows (3x12)', 'equipment': ['cable machine']},
                {'exercise': 'Dumbbell Rows (3x12)', 'equipment': ['dumbbells']},
                {'exercise': 'Inverted Rows (3x10)', 'equipment': ['bodyweight only']}
            ],
            'intermediate': [
                {'exercise': 'Pull-ups (4x8)', 'equipment': ['bodyweight only', 'full gym']},
                {'exercise': 'Barbell Rows (4x10)', 'equipment': ['barbell']},
                {'exercise': 'T-Bar Rows (3x12)', 'equipment': ['full gym']},
                {'exercise': 'Face Pulls (3x15)', 'equipment': ['cable machine']},
                {'exercise': 'Meadows Row (3x10)', 'equipment': ['barbell']}
            ],
            'advanced': [
                {'exercise': 'Weighted Pull-ups (4x8)', 'equipment': ['full gym']},
                {'exercise': 'Pendlay Rows (4x8)', 'equipment': ['barbell']},
                {'exercise': 'Single-Arm Dumbbell Rows (4x10)', 'equipment': ['dumbbells']},
                {'exercise': 'Rack Pulls (4x6)', 'equipment': ['barbell']},
                {'exercise': 'Cable Pullovers (3x12)', 'equipment': ['cable machine']}
            ]
        },
        'legs': {
            'beginner': [
                {'exercise': 'Bodyweight Squats (3x15)', 'equipment': ['bodyweight only']},
                {'exercise': 'Dumbbell Lunges (3x10/leg)', 'equipment': ['dumbbells']},
                {'exercise': 'Leg Press (3x12)', 'equipment': ['full gym']},
                {'exercise': 'Leg Extensions (3x15)', 'equipment': ['full gym']},
                {'exercise': 'Goblet Squats (3x12)', 'equipment': ['dumbbells']}
            ],
            'intermediate': [
                {'exercise': 'Barbell Squats (4x8)', 'equipment': ['barbell']},
                {'exercise': 'Romanian Deadlifts (3x10)', 'equipment': ['barbell']},
                {'exercise': 'Walking Lunges (3x12/leg)', 'equipment': ['dumbbells']},
                {'exercise': 'Bulgarian Split Squats (3x10/leg)', 'equipment': ['dumbbells']},
                {'exercise': 'Cable Pull Throughs (3x12)', 'equipment': ['cable machine']}
            ],
            'advanced': [
                {'exercise': 'Front Squats (4x6)', 'equipment': ['barbell']},
                {'exercise': 'Hack Squats (4x8)', 'equipment': ['full gym']},
                {'exercise': 'Single-Leg Press (3x10/leg)', 'equipment': ['full gym']},
                {'exercise': 'Walking Lunges (4x12/leg)', 'equipment': ['barbell']},
                {'exercise': 'Sissy Squats (3x12)', 'equipment': ['bodyweight only']}
            ]
        },
        'shoulders': {
            'beginner': [
                {'exercise': 'Dumbbell Shoulder Press (3x10)', 'equipment': ['dumbbells']},
                {'exercise': 'Lateral Raises (3x12)', 'equipment': ['dumbbells']},
                {'exercise': 'Front Raises (3x12)', 'equipment': ['dumbbells']},
                {'exercise': 'Machine Shoulder Press (3x12)', 'equipment': ['full gym']},
                {'exercise': 'Pike Push-ups (3x8)', 'equipment': ['bodyweight only']}
            ],
            'intermediate': [
                {'exercise': 'Military Press (4x8)', 'equipment': ['barbell']},
                {'exercise': 'Face Pulls (3x15)', 'equipment': ['cable machine']},
                {'exercise': 'Upright Rows (3x12)', 'equipment': ['barbell', 'dumbbells']},
                {'exercise': 'Cable Lateral Raises (3x12)', 'equipment': ['cable machine']},
                {'exercise': 'Arnold Press (3x10)', 'equipment': ['dumbbells']}
            ],
            'advanced': [
                {'exercise': 'Push Press (4x6)', 'equipment': ['barbell']},
                {'exercise': 'Behind the Neck Press (4x8)', 'equipment': ['barbell']},
                {'exercise': 'Single-Arm Cable Laterals (4x12)', 'equipment': ['cable machine']},
                {'exercise': 'Handstand Push-ups (3x8)', 'equipment': ['bodyweight only']},
                {'exercise': 'Viking Press (3x10)', 'equipment': ['full gym']}
            ]
        },
        'arms': {
            'beginner': [
                {'exercise': 'Dumbbell Bicep Curls (3x12)', 'equipment': ['dumbbells']},
                {'exercise': 'Tricep Pushdowns (3x12)', 'equipment': ['cable machine']},
                {'exercise': 'Hammer Curls (3x12)', 'equipment': ['dumbbells']},
                {'exercise': 'Diamond Push-ups (3x10)', 'equipment': ['bodyweight only']},
                {'exercise': 'Machine Curls (3x12)', 'equipment': ['full gym']}
            ],
            'intermediate': [
                {'exercise': 'Barbell Curls (4x10)', 'equipment': ['barbell']},
                {'exercise': 'Skull Crushers (3x12)', 'equipment': ['barbell', 'dumbbells']},
                {'exercise': 'Incline Dumbbell Curls (3x12)', 'equipment': ['dumbbells']},
                {'exercise': 'Rope Pushdowns (3x15)', 'equipment': ['cable machine']},
                {'exercise': 'Preacher Curls (3x12)', 'equipment': ['full gym', 'barbell']}
            ],
            'advanced': [
                {'exercise': 'Close-Grip Bench Press (4x8)', 'equipment': ['barbell']},
                {'exercise': '21s Bicep Curls (3x21)', 'equipment': ['barbell', 'dumbbells']},
                {'exercise': 'Weighted Dips (4x10)', 'equipment': ['full gym']},
                {'exercise': 'Spider Curls (3x12)', 'equipment': ['dumbbells']},
                {'exercise': 'Overhead Tricep Extensions (4x12)', 'equipment': ['cable machine']}
            ]
        },
        'core': {
            'beginner': [
                {'exercise': 'Plank (3x30s)', 'equipment': ['bodyweight only']},
                {'exercise': 'Crunches (3x15)', 'equipment': ['bodyweight only']},
                {'exercise': 'Russian Twists (3x15)', 'equipment': ['bodyweight only']},
                {'exercise': 'Dead Bug (3x10/side)', 'equipment': ['bodyweight only']},
                {'exercise': 'Cable Crunches (3x15)', 'equipment': ['cable machine']}
            ],
            'intermediate': [
                {'exercise': 'Hanging Leg Raises (3x12)', 'equipment': ['full gym', 'bodyweight only']},
                {'exercise': 'Cable Woodchoppers (3x10/side)', 'equipment': ['cable machine']},
                {'exercise': 'Decline Bench Crunches (3x15)', 'equipment': ['full gym']},
                {'exercise': 'Plank with Shoulder Taps (3x10/side)', 'equipment': ['bodyweight only']},
                {'exercise': 'Side Planks (3x30s/side)', 'equipment': ['bodyweight only']}
            ],
            'advanced': [
                {'exercise': 'Dragon Flags (3x8)', 'equipment': ['full gym', 'bodyweight only']},
                {'exercise': 'Ab Wheel Rollouts (3x10)', 'equipment': ['full gym']},
                {'exercise': 'Weighted Hanging Leg Raises (4x12)', 'equipment': ['full gym']},
                {'exercise': 'Cable Crunches (4x15)', 'equipment': ['cable machine']},
                {'exercise': 'L-Sits (3x20s)', 'equipment': ['bodyweight only']}
            ]
        },
        'full body': {
            'beginner': [
                {'exercise': 'Push-ups (3x10)', 'equipment': ['bodyweight only']},
                {'exercise': 'Bodyweight Squats (3x12)', 'equipment': ['bodyweight only']},
                {'exercise': 'Dumbbell Rows (3x12)', 'equipment': ['dumbbells']},
                {'exercise': 'Lunges (3x10/leg)', 'equipment': ['bodyweight only']},
                {'exercise': 'Plank (3x30s)', 'equipment': ['bodyweight only']}
            ],
            'intermediate': [
                {'exercise': 'Barbell Squats (4x8)', 'equipment': ['barbell']},
                {'exercise': 'Pull-ups (3x8)', 'equipment': ['bodyweight only', 'full gym']},
                {'exercise': 'Dumbbell Press (4x10)', 'equipment': ['dumbbells']},
                {'exercise': 'Romanian Deadlifts (3x10)', 'equipment': ['barbell']},
                {'exercise': 'Military Press (3x10)', 'equipment': ['barbell']}
            ],
            'advanced': [
                {'exercise': 'Deadlifts (5x5)', 'equipment': ['barbell']},
                {'exercise': 'Clean and Press (4x6)', 'equipment': ['barbell']},
                {'exercise': 'Weighted Pull-ups (4x8)', 'equipment': ['full gym']},
                {'exercise': 'Front Squats (4x8)', 'equipment': ['barbell']},
                {'exercise': 'Dips (4x10)', 'equipment': ['bodyweight only', 'full gym']}
            ]
        }
    }

    # get exercises for the specified muscle group and experience level
    available_exercises = workouts.get(muscle_group, {}).get(experience_level, [])
    
    if not available_exercises:
        raise WorkoutGenerationError(f"No exercises found for {muscle_group} at {experience_level} level")

    # if full gym is selected, allow all exercises
    if 'full gym' in equipment_available:
        selected_exercises = [ex['exercise'] for ex in available_exercises]
    else:
        # filter exercises based on available equipment
        selected_exercises = [
            ex['exercise'] for ex in available_exercises 
            if any(eq in equipment_available for eq in ex['equipment'])
        ]

    # if no exercises match the equipment, provide bodyweight alternatives
    if not selected_exercises:
        bodyweight_alternatives = {
            'chest': [
                'Push-ups (3x12)',
                'Diamond Push-ups (3x10)',
                'Wide Push-ups (3x10)',
                'Decline Push-ups (3x10)'
            ],
            'back': [
                'Inverted Rows (3x10)',
                'Superman Holds (3x30s)',
                'Band Pull-aparts (3x15)',
                'Wall Slides (3x12)'
            ],
            'legs': [
                'Bodyweight Squats (3x15)',
                'Walking Lunges (3x10/leg)',
                'Jump Squats (3x10)',
                'Glute Bridges (3x15)'
            ],
            'shoulders': [
                'Pike Push-ups (3x8)',
                'Wall Handstand Holds (3x30s)',
                'Arm Circles (3x20)',
                'Decline Push-ups (3x10)'
            ],
            'arms': [
                'Diamond Push-ups (3x12)',
                'Close-grip Push-ups (3x10)',
                'Bench Dips (3x12)',
                'Inverted Row Curls (3x10)'
            ],
            'core': [
                'Plank (3x30s)',
                'Mountain Climbers (3x20)',
                'Russian Twists (3x15)',
                'Leg Raises (3x12)'
            ],
            'full body': [
                'Burpees (3x10)',
                'Mountain Climbers (3x20)',
                'Push-ups (3x10)',
                'Bodyweight Squats (3x15)'
            ]
        }
        
        selected_exercises = bodyweight_alternatives.get(muscle_group, [
            'Push-ups (3x10)',
            'Squats (3x15)',
            'Planks (3x30s)'
        ])
        
        return selected_exercises

    # return 3-4 exercises, with a note if using bodyweight alternatives
    if len(selected_exercises) < 3:
        raise WorkoutGenerationError(
            f"Insufficient exercises found for {muscle_group} with available equipment. "
            "Consider adding more equipment or using bodyweight alternatives."
        )

    return selected_exercises[:4]  # limit to 4 exercises maximum

def calculate_calories(age: float, weight: float, height: float, gender: str, activity_level: str, goal: str) -> int:
    """
    Calculate daily calorie needs based on user metrics and goals.
    
    Args:
        age (float): Age in years
        weight (float): Weight in kg
        height (float): Height in cm
        gender (str): 'Male' or 'Female'
        activity_level (str): Activity level from predefined options
        goal (str): Weight goal (Maintain/Lose Weight/Gain Weight)
    
    Returns:
        int: Daily calorie recommendation
    """
    # calculate Basal Metabolic Rate (BMR) using Mifflin-St Jeor Equation
    if gender.lower() == 'male':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    # activity level multipliers (updated with more precise values)
    activity_multipliers = {
        'sedentary': 1.2,
        'light': 1.375,
        'moderate': 1.55,
        'very active': 1.725,
        'extra active': 1.9
    }

    # get the correct multiplier based on activity level (case-insensitive)
    activity_level = activity_level.lower()
    multiplier = activity_multipliers.get(activity_level, 1.2)  # default to sedentary if invalid

    # calculate Total Daily Energy Expenditure (TDEE)
    tdee = bmr * multiplier

    # adjust calories based on goal
    goal = goal.lower()
    if goal == 'lose weight':
        # create a moderate caloric deficit (500 calories for approximately 0.5kg/week loss)
        calories = tdee - 500
    elif goal == 'gain weight':
        # create a moderate caloric surplus (500 calories for approximately 0.5kg/week gain)
        calories = tdee + 500
    else:  # maintain weight
        calories = tdee

    # round to nearest 50 calories for more practical recommendations
    return round(calories / 50) * 50

def format_calorie_response(calories: int, goal: str, activity_level: str) -> str:
    """
    Format the calorie calculation response with additional context and recommendations.
    
    Args:
        calories (int): Calculated daily calories
        goal (str): User's goal
        activity_level (str): User's activity level
    
    Returns:
        str: Formatted response with recommendations
    """
    response = [f"🎯 **Your Daily Calorie Target: {calories} calories**\n"]
    
    # add goal-specific advice
    if goal.lower() == 'lose weight':
        response.extend([
            "📉 **Weight Loss Breakdown:**",
            f"- This represents a 500 calorie deficit",
            "- Expected weight loss: ~0.5kg (1lb) per week",
            "- Focus on protein intake (1.6-2.2g per kg of body weight)",
            "- Include plenty of vegetables for satiety"
        ])
    elif goal.lower() == 'gain weight':
        response.extend([
            "📈 **Weight Gain Breakdown:**",
            f"- This represents a 500 calorie surplus",
            "- Expected weight gain: ~0.5kg (1lb) per week",
            "- Prioritize protein (1.6-2.2g per kg of body weight)",
            "- Include complex carbs for energy"
        ])
    else:
        response.extend([
            "⚖️ **Maintenance Breakdown:**",
            "- This maintains your current weight",
            "- Great for body recomposition",
            "- Ideal for performance goals"
        ])

    # add activity level context
    response.append(f"\n💪 **Activity Level ({activity_level}):**")
    activity_context = {
        'sedentary': "Mostly sitting throughout the day with little exercise",
        'light': "Light exercise/sports 1-3 days/week",
        'moderate': "Moderate exercise/sports 3-5 days/week",
        'very active': "Hard exercise/sports 6-7 days/week",
        'extra active': "Very hard exercise/sports & physical job or training twice per day"
    }
    response.append(f"- {activity_context.get(activity_level.lower(), 'Custom activity level')}")

    # add general recommendations
    response.extend([
        "\n📋 **General Recommendations:**",
        "- Track your calories and weight for 2-3 weeks",
        "- Adjust intake based on actual results",
        "- Stay hydrated (2-3 liters of water daily)",
        "- Get 7-9 hours of sleep",
        "- Consider using a food tracking app"
    ])

    return "\n".join(response)

@dataclass
class Supplement:
    name: str
    dosage: str
    benefits: List[str]
    dietary_restrictions: Set[str]
    warning: str = ""

def recommend_supplements(goal: str, diet_preferences: List[str]) -> str:
    """
    Recommend supplements based on fitness goals and dietary preferences.
    
    Args:
        goal (str): Fitness goal (Muscle Gain/Fat Loss/Performance/General Health)
        diet_preferences (List[str]): List of dietary preferences/restrictions
    
    Returns:
        str: Formatted supplement recommendations with dosages and explanations
    """
    # convert inputs to lowercase for consistency
    goal = goal.lower()
    diet_preferences = [pref.lower() for pref in diet_preferences]

    # define supplement database with dietary restrictions
    supplements_db = {
        'muscle gain': [
            Supplement(
                name="Creatine Monohydrate",
                dosage="5g daily",
                benefits=[
                    "Increases muscle strength and power",
                    "Improves muscle recovery",
                    "Enhances muscle growth"
                ],
                dietary_restrictions=set(),
            ),
            Supplement(
                name="Whey Protein",
                dosage="20-30g post-workout",
                benefits=[
                    "Fast-absorbing protein",
                    "Rich in BCAAs",
                    "Supports muscle recovery"
                ],
                dietary_restrictions={'vegan', 'lactose-free'},
            ),
            Supplement(
                name="Pea Protein",
                dosage="20-30g post-workout",
                benefits=[
                    "Plant-based protein source",
                    "Good amino acid profile",
                    "Easy to digest"
                ],
                dietary_restrictions=set(),
            ),
            Supplement(
                name="Beta-Alanine",
                dosage="3-5g daily",
                benefits=[
                    "Reduces muscle fatigue",
                    "Improves exercise performance",
                    "Increases muscle endurance"
                ],
                dietary_restrictions=set(),
            )
        ],
        'fat loss': [
            Supplement(
                name="Protein Powder (Low-carb)",
                dosage="20-30g as needed",
                benefits=[
                    "Preserves muscle mass",
                    "Increases satiety",
                    "Supports recovery"
                ],
                dietary_restrictions={'vegan'},
            ),
            Supplement(
                name="Green Tea Extract",
                dosage="300-500mg daily",
                benefits=[
                    "Supports metabolism",
                    "Contains antioxidants",
                    "May aid fat oxidation"
                ],
                dietary_restrictions=set(),
            ),
            Supplement(
                name="L-Carnitine",
                dosage="2-3g daily",
                benefits=[
                    "Supports fat metabolism",
                    "May improve exercise performance",
                    "Can reduce fatigue"
                ],
                dietary_restrictions=set(),
            )
        ],
        'performance': [
            Supplement(
                name="Creatine Monohydrate",
                dosage="5g daily",
                benefits=[
                    "Improves power output",
                    "Enhances high-intensity performance",
                    "Reduces fatigue"
                ],
                dietary_restrictions=set(),
            ),
            Supplement(
                name="Beta-Alanine",
                dosage="3-5g daily",
                benefits=[
                    "Improves endurance",
                    "Reduces muscle fatigue",
                    "Enhances performance in high-intensity exercises"
                ],
                dietary_restrictions=set(),
            ),
            Supplement(
                name="Caffeine",
                dosage="200-400mg pre-workout",
                benefits=[
                    "Increases alertness",
                    "Improves power output",
                    "Enhances endurance"
                ],
                dietary_restrictions=set(),
            ),
            Supplement(
                name="Essential Amino Acids (EAAs)",
                dosage="5-10g during workout",
                benefits=[
                    "Supports muscle recovery",
                    "Reduces fatigue",
                    "Maintains muscle mass"
                ],
                dietary_restrictions=set(),
            ),
            Supplement(
                name="Fish Oil",
                dosage="2-3g daily",
                benefits=[
                    "Reduces inflammation",
                    "Supports joint health",
                    "Improves recovery"
                ],
                dietary_restrictions={'vegan', 'vegetarian'},
            ),
            Supplement(
                name="Algae Omega-3",
                dosage="2-3g daily",
                benefits=[
                    "Plant-based omega-3 source",
                    "Supports recovery",
                    "Anti-inflammatory properties"
                ],
                dietary_restrictions=set(),
            )
        ],
        'general health': [
            Supplement(
                name="Multivitamin",
                dosage="As directed on label",
                benefits=[
                    "Fills nutritional gaps",
                    "Supports overall health",
                    "Enhances recovery"
                ],
                dietary_restrictions=set(),
            ),
            Supplement(
                name="Vitamin D3",
                dosage="1000-5000 IU daily",
                benefits=[
                    "Supports bone health",
                    "Enhances immune function",
                    "Improves muscle function"
                ],
                dietary_restrictions={'vegan'},
            ),
            Supplement(
                name="Vegan Vitamin D3",
                dosage="1000-5000 IU daily",
                benefits=[
                    "Plant-based vitamin D source",
                    "Supports bone health",
                    "Enhances immune function"
                ],
                dietary_restrictions=set(),
            ),
            Supplement(
                name="Fish Oil",
                dosage="2-3g daily",
                benefits=[
                    "Supports heart health",
                    "Reduces inflammation",
                    "Improves brain function"
                ],
                dietary_restrictions={'vegan', 'vegetarian'},
            ),
            Supplement(
                name="Algae Omega-3",
                dosage="2-3g daily",
                benefits=[
                    "Plant-based omega-3 source",
                    "Supports heart health",
                    "Improves brain function"
                ],
                dietary_restrictions=set(),
            ),
            Supplement(
                name="Probiotics",
                dosage="1-2 capsules daily",
                benefits=[
                    "Supports gut health",
                    "Enhances immune function",
                    "Improves nutrient absorption"
                ],
                dietary_restrictions=set(),
            )
        ]
    }

    # get supplements for the specified goal
    available_supplements = supplements_db.get(goal, supplements_db['general health'])

    # filter supplements based on dietary preferences
    filtered_supplements = []
    for supp in available_supplements:
        # skip if supplement conflicts with any dietary preference
        if any(pref in supp.dietary_restrictions for pref in diet_preferences):
            continue
        filtered_supplements.append(supp)

    # add alternative supplements based on dietary preferences
    if 'vegan' in diet_preferences or 'vegetarian' in diet_preferences:
        if goal in ['muscle gain', 'fat loss']:
            filtered_supplements.append(
                Supplement(
                    name="Pea Protein",
                    dosage="20-30g as needed",
                    benefits=[
                        "Complete plant-based protein",
                        "Rich in iron",
                        "Excellent amino acid profile"
                    ],
                    dietary_restrictions=set(),
                )
            )

    # format the response
    if not filtered_supplements:
        return "No supplements found matching your dietary preferences. Please consult a nutritionist for personalized advice."

    response = [f"🎯 **Recommended Supplements for {goal.title()}**\n"]
    response.append("*Filtered for your dietary preferences*\n")

    for supp in filtered_supplements:
        response.append(f"**{supp.name}**")
        response.append(f"- Dosage: {supp.dosage}")
        response.append("- Benefits:")
        for benefit in supp.benefits:
            response.append(f"  • {benefit}")
        if supp.warning:
            response.append(f"- ⚠️ *{supp.warning}*")
        response.append("")

    response.extend([
        "**Important Notes:**",
        "- Always consult with a healthcare provider before starting any supplement regimen",
        "- Start with lower doses to assess tolerance",
        "- Quality matters - choose reputable brands",
        "- Supplements are not a replacement for a balanced diet",
        "- Store supplements as directed on the label"
    ])

    return "\n".join(response)

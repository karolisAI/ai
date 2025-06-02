import re
from typing import Dict, Any, Union, List
import logging

def sanitize_numeric_input(value: str) -> Union[float, None]:
    """Sanitize and validate numeric inputs for measurements"""
    try:
        num = float(value)
        if num <= 0:
            return None
        return num
    except ValueError:
        return None

def validate_health_metrics(metrics: Dict[str, Any]) -> List[str]:
    """Validate health-related metrics for safety and reasonability"""
    errors = []
    
    # age validation
    if 'age' in metrics:
        age = sanitize_numeric_input(str(metrics['age']))
        if not age or age < 13 or age > 120:
            errors.append("Age must be between 13 and 120 years")
    
    # weight validation (in kg)
    if 'weight' in metrics:
        weight = sanitize_numeric_input(str(metrics['weight']))
        if not weight or weight < 30 or weight > 300:
            errors.append("Weight must be between 30 and 300 kg")
    
    # height validation (in cm)
    if 'height' in metrics:
        height = sanitize_numeric_input(str(metrics['height']))
        if not height or height < 120 or height > 250:
            errors.append("Height must be between 120 and 250 cm")
    
    return errors

def sanitize_workout_input(input_str: str) -> str:
    """Sanitize workout-related input for safety"""
    # remove any potentially harmful characters
    sanitized = re.sub(r'[^a-zA-Z0-9\s\-_]', '', input_str)
    return sanitized.strip()

def validate_exercise_safety(exercise_type: str, experience_level: str) -> bool:
    """Validate if an exercise is appropriate for the user's experience level"""
    high_risk_exercises = {
        'deadlifts': ['beginner'],
        'clean_and_jerk': ['beginner'],
        'snatch': ['beginner'],
        'muscle_ups': ['beginner'],
        'handstand_pushups': ['beginner']
    }
    
    exercise_type = exercise_type.lower().replace(' ', '_')
    if exercise_type in high_risk_exercises:
        return experience_level.lower() not in high_risk_exercises[exercise_type]
    return True

def validate_supplement_input(supplement_data: Dict[str, Any]) -> List[str]:
    """Validate supplement-related inputs for safety"""
    warnings = []
    
    # list of supplements that require medical consultation
    medical_consultation_required = [
        'sarms', 'steroids', 'hormones', 'prohormones', 'testosterone',
        'growth hormone', 'peptides'
    ]
    
    if 'goal' in supplement_data:
        goal = supplement_data['goal'].lower()
        if any(term in goal for term in medical_consultation_required):
            warnings.append("Medical consultation required for performance-enhancing substances")
    
    return warnings

def log_security_event(event_type: str, details: Dict[str, Any]) -> None:
    """Log security-related events"""
    logging.warning(f"Security Event - Type: {event_type}, Details: {details}")

def validate_user_age_consent(age: int) -> bool:
    """Validate user age for consent and safety"""
    return age >= 13  # minimum age for fitness advice

def check_health_warning_conditions(metrics: Dict[str, Any]) -> List[str]:
    """Check for conditions that require health warnings"""
    warnings = []
    
    # BMI calculation and warning
    if 'weight' in metrics and 'height' in metrics:
        weight = float(metrics['weight'])
        height = float(metrics['height']) / 100  # convert cm to m
        bmi = weight / (height * height)
        
        if bmi < 16:
            warnings.append("WARNING: BMI indicates severe underweight. Please consult a healthcare provider.")
        elif bmi > 35:
            warnings.append("WARNING: BMI indicates obesity. Please consult a healthcare provider.")
    
    # rapid weight change warning
    if 'goal' in metrics:
        if 'lose' in metrics['goal'].lower():
            warnings.append("Note: Healthy weight loss should not exceed 1kg per week.")
        elif 'gain' in metrics['goal'].lower():
            warnings.append("Note: Healthy weight gain should not exceed 0.5kg per week.")
    
    return warnings 
// Set today's date as default
document.getElementById('travelDate').valueAsDate = new Date();

// Form submission handler
document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = new FormData(this);
    const data = {
        train_number: formData.get('train_number'),
        start_station: formData.get('start_station'),
        arrive_station: formData.get('arrive_station'),
        travel_date: formData.get('travel_date')
    };

    // Validate form
    if (!data.train_number || !data.start_station || !data.arrive_station || !data.travel_date) {
        showError('Please fill in all fields');
        return;
    }

    if (data.start_station === data.arrive_station) {
        showError('Start and arrival stations cannot be the same');
        return;
    }

    // Show loading, hide results and errors
    showLoading(true);
    hideAllMessages();
    document.getElementById('resultContainer').classList.remove('show');
    document.getElementById('initialMessage').style.display = 'none';

    try {
        console.log('📤 Sending request:', data);
        
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(data)
        });

        console.log('📥 Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Server error (${response.status}): ${errorText}`);
        }

        const text = await response.text();
        console.log('📥 Raw response:', text);
        
        let result;
        try {
            result = JSON.parse(text);
        } catch (parseError) {
            console.error('❌ JSON parse error:', parseError);
            throw new Error('Invalid JSON response from server. Please try again.');
        }

        console.log('✅ Parsed result:', result);

        showLoading(false);

        if (result.success) {
            showSuccess('Prediction completed successfully!');
            displayResults(result);
        } else {
            showError(result.error || 'Prediction failed. Please try again.');
        }

    } catch (error) {
        showLoading(false);
        console.error('❌ Network error:', error);
        showError('Network error: ' + error.message);
    }
});

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
    document.getElementById('predictBtn').disabled = show;
    if (show) {
        document.getElementById('predictBtn').innerHTML = '⏳ Processing...';
    } else {
        document.getElementById('predictBtn').innerHTML = '🚀 Predict Journey Times';
    }
}

function showError(message) {
    hideAllMessages();
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorContainer').style.display = 'block';
}

function showSuccess(message) {
    hideAllMessages();
    document.getElementById('successText').textContent = message;
    document.getElementById('successMessage').style.display = 'block';
    
    // Hide success message after 3 seconds
    setTimeout(() => {
        document.getElementById('successMessage').style.display = 'none';
    }, 3000);
}

function hideAllMessages() {
    document.getElementById('errorContainer').style.display = 'none';
    document.getElementById('successMessage').style.display = 'none';
}

function displayResults(result) {
    console.log('🔄 Displaying results:', result);
    
    // Safely update journey info with null checks
    document.getElementById('journeyRoute').textContent = 
        `${result.input?.start_station || '--'} to ${result.input?.arrive_station || '--'}`;
    document.getElementById('journeyTrain').textContent = `Train ${result.input?.train_number || '--'}`;
    document.getElementById('journeyDate').textContent = result.input?.travel_date || '--';

    // Safely update weather information with null checks
    const weather = result.weather || {};
    document.getElementById('weatherStation').textContent = `Weather at ${weather.station || '--'}`;
    
    // Safe weather condition handling
    const weatherCondition = weather.description || weather.condition || 'No data';
    document.getElementById('weatherCondition').textContent = 
        typeof weatherCondition === 'string' ? 
        weatherCondition.charAt(0).toUpperCase() + weatherCondition.slice(1) : 
        'No data';
    
    document.getElementById('weatherTemp').textContent = weather.temperature || '--';
    document.getElementById('weatherTempMain').textContent = weather.temperature ? `${Math.round(weather.temperature)}°` : '--°';
    document.getElementById('weatherWind').textContent = weather.wind_speed || '--';
    document.getElementById('weatherPrecip').textContent = weather.precipitation || '--';
    document.getElementById('weatherVisibility').textContent = weather.visibility || '--';
    document.getElementById('weatherHumidity').textContent = weather.humidity || '--';
    document.getElementById('weatherMain').textContent = weather.main_condition || '--';

    // Update weather icon based on condition
    const weatherIcon = getWeatherIcon(weather.main_condition || weather.condition);
    document.getElementById('weatherIcon').textContent = weatherIcon;

    // Safely update timetable and predictions
    const timetable = result.timetable || {};
    const prediction = result.prediction || {};
    
    document.getElementById('plannedDeparture').textContent = timetable.planned_departure || '--:--';
    document.getElementById('plannedArrival').textContent = timetable.planned_arrival || '--:--';
    document.getElementById('predictedDeparture').textContent = prediction.predicted_departure || '--:--';
    document.getElementById('predictedArrival').textContent = prediction.predicted_arrival || '--:--';
    
    document.getElementById('journeyDistance').textContent = `${timetable.distance_km || '--'} km`;
    document.getElementById('travelTime').textContent = `${Math.round(prediction.predicted_travel_time_minutes) || '--'} min`;

    // Safely update differences
    const depDiff = prediction.departure_difference_minutes || 0;
    const arrDiff = prediction.arrival_difference_minutes || 0;
    
    document.getElementById('departureDifference').textContent = formatDifference(depDiff);
    document.getElementById('departureDifference').className = `difference ${getDifferenceClass(depDiff)}`;
    
    document.getElementById('arrivalDifference').textContent = formatDifference(arrDiff);
    document.getElementById('arrivalDifference').className = `difference ${getDifferenceClass(arrDiff)}`;

    // Safely update reliability
    const reliability = result.reliability || {};
    document.getElementById('departureConfidence').textContent = reliability.departure_rating || '--';
    document.getElementById('arrivalConfidence').textContent = reliability.arrival_rating || '--';
    document.getElementById('weatherImpact').textContent = reliability.weather_impact || '--';
    document.getElementById('overallConfidence').textContent = reliability.overall_confidence || '--';

    if (reliability.notes && Array.isArray(reliability.notes)) {
        document.getElementById('reliabilityNotes').innerHTML = 
            reliability.notes.map(note => `• ${note}`).join('<br>');
    } else {
        document.getElementById('reliabilityNotes').textContent = 'Using real-time weather data from your selected travel date';
    }

    // Show model info
    document.getElementById('modelUsed').textContent = result.model_used || 'AI Model';

    // Show results
    document.getElementById('resultContainer').classList.add('show');
    document.getElementById('initialMessage').style.display = 'none';
    
    console.log('✅ Results displayed successfully');
}

function getWeatherIcon(condition) {
    if (!condition) return '🌤️';
    
    const conditionLower = String(condition).toLowerCase();
    if (conditionLower.includes('rain') || conditionLower.includes('drizzle')) return '🌧️';
    if (conditionLower.includes('cloud')) return '☁️';
    if (conditionLower.includes('clear') || conditionLower.includes('sunny')) return '☀️';
    if (conditionLower.includes('snow')) return '❄️';
    if (conditionLower.includes('fog') || conditionLower.includes('mist')) return '🌫️';
    if (conditionLower.includes('thunderstorm')) return '⛈️';
    return '🌤️';
}

function formatDifference(diff) {
    const numDiff = Number(diff) || 0;
    if (numDiff > 0) return `+${Math.round(numDiff)} min delay`;
    if (numDiff < 0) return `${Math.round(numDiff)} min early`;
    return 'On time';
}

function getDifferenceClass(diff) {
    const numDiff = Number(diff) || 0;
    if (numDiff > 5) return 'positive';
    if (numDiff < -5) return 'negative';
    return 'neutral';
}

// Prevent selecting same station for start and arrival
function updateStationOptions() {
    const startSelect = document.getElementById('startStation');
    const arriveSelect = document.getElementById('arriveStation');
    const selectedStart = startSelect.value;
    const selectedArrive = arriveSelect.value;

    // Enable all options first
    Array.from(startSelect.options).forEach(option => {
        if (option.value) {
            option.disabled = false;
            option.style.color = '';
        }
    });
    Array.from(arriveSelect.options).forEach(option => {
        if (option.value) {
            option.disabled = false;
            option.style.color = '';
        }
    });

    // Disable selected station in opposite dropdown
    if (selectedStart) {
        Array.from(arriveSelect.options).forEach(option => {
            if (option.value === selectedStart) {
                option.disabled = true;
                option.style.color = '#ccc';
            }
        });
    }
    if (selectedArrive) {
        Array.from(startSelect.options).forEach(option => {
            if (option.value === selectedArrive) {
                option.disabled = true;
                option.style.color = '#ccc';
            }
        });
    }
}

document.getElementById('startStation').addEventListener('change', updateStationOptions);
document.getElementById('arriveStation').addEventListener('change', updateStationOptions);

// Test connection on page load
window.addEventListener('load', function() {
    console.log('🔍 Testing backend connection...');
    fetch('/api/health')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('✅ Backend health:', data);
            if (!data.success) {
                console.warn('⚠️ Backend reported issues:', data);
            }
        })
        .catch(error => {
            console.error('❌ Backend connection failed:', error);
            showError('Backend connection failed. Please make sure the server is running.');
        });
    
    // Initialize station options
    updateStationOptions();
});

// Add input validation
document.querySelectorAll('input, select').forEach(element => {
    element.addEventListener('invalid', function(e) {
        e.preventDefault();
        showError('Please fill in all required fields correctly.');
    });
});
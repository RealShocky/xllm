document.addEventListener('DOMContentLoaded', function() {
    const queryForm = document.getElementById('queryForm');
    const responsesDiv = document.getElementById('responses');

    queryForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        const prompt = document.getElementById('prompt').value;

        const response = await fetch('/get_responses', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompts: [prompt] }),
        });

        const responses = await response.json();
        responsesDiv.innerHTML = '';

        responses.forEach((res, index) => {
            const responseElem = document.createElement('div');
            responseElem.classList.add('response');
            responseElem.innerText = `Response ${index + 1}: ${res}`;
            responsesDiv.appendChild(responseElem);
        });
    });
});

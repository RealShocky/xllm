document.addEventListener('DOMContentLoaded', function() {
    const xllm6Form = document.getElementById('xllm6Form');
    const resultsDiv = document.getElementById('results');

    xllm6Form.addEventListener('submit', async function(event) {
        event.preventDefault();
        const query = document.getElementById('queryInput').value;

        const response = await fetch('/xllm6_process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query }),
        });

        const result = await response.json();
        resultsDiv.innerHTML = `<pre>${JSON.stringify(result, null, 2)}</pre>`;
    });
});

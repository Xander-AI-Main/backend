
const url = "https://api.xanderco.in/core/interference/";

const data = {
    data: ['4.7', '8.0', '64.0', '6.1', '35.0', '12.0', '3800.0'],
    modelId: '1e367096-849a-41fe-9419-41a40f7e4ad7',
    userId: '22',
};

const headers = {
    'Content-Type': 'application/json',
};

fetch(url, {
    method: 'POST',
    headers: headers,
    body: JSON.stringify(data)
})
.then(response => response.json().then(data => {
    if (response.ok) {
        console.log("Response:");
        console.log(JSON.stringify(data, null, 2));
    } else {
        console.error(`Error: ${response.status}`);
        console.error(data);
    }
}))
.catch(error => {
    console.error(`An error occurred: ${error}`);
});

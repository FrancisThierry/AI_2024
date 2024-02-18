var http = require('http'); 

var fs = require('fs'); 

  

const PORT = 8080; // Spécifiez un numéro de port (par exemple, 3000, 8080) 

  

fs.readFile('./index.html', function (err, html) { 

    if (err) throw err; 

  

    http.createServer(function (request, response) { 

        response.writeHead(200, { "Content-Type": "text/html" }); 

        response.write(html); 

        response.end(); 

    }).listen(PORT); 

}); 

console.log(`Serveur à l'écoute sur le port ${PORT}`); 
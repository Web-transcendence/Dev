import fastify from 'fastify'
import {addUser} from "./addUser.js";
import {checkToken, checkUser} from "./CheckToken.js";
import {googleAuth} from "./googleApi.js";


const app = fastify();

app.post('/sign-up', addUser);
app.post('/check-token', checkToken);
app.post('/user-login', checkUser);
app.post('/auth/google', googleAuth);

app.listen({port: 5000, host: '0.0.0.0'}, (err, adrr) => {
    if (err) {
        console.error(err);
        process.exit(1);
    }
    console.log(`server running on ${adrr}`)
})
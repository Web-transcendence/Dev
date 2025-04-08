"use strict";
/* ************************************************************************** */
/*                                                                            */
/*                                                        :::      ::::::::   */
/*   server.ts                                          :+:      :+:    :+:   */
/*                                                    +:+ +:+         +:+     */
/*   By: thibaud <thibaud@student.42.fr>            +#+  +:+       +#+        */
/*                                                +#+#+#+#+#+   +#+           */
/*   Created: 2025/04/08 09:53:35 by thibaud           #+#    #+#             */
/*   Updated: 2025/04/08 11:18:22 by thibaud          ###   ########.fr       */
/*                                                                            */
/* ************************************************************************** */
Object.defineProperty(exports, "__esModule", { value: true });
var ws_1 = require("ws");
var wss = new ws_1.WebSocketServer({ port: 8080 });
wss.on('connection', function (ws) {
    console.log('Client Connected');
    ws.on('message', function (message) {
        var aligned = Buffer.from(message);
        var doubleArray = new Float64Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 8);
        console.log("Received:  ", Array.from(doubleArray));
        var float64 = new Float64Array(3);
        float64[0] = 0.2;
        float64[1] = 21.2;
        float64[2] = 3243.3;
        ws.send(float64);
    });
    ws.on('close', function () {
        console.log('Client disconnected');
    });
});
console.log('WebSocket server running on ws://localhost:8080');

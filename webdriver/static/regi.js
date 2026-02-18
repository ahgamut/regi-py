// websocket functions
function init_ws() {
    let g = Alpine.store('gamestate');
    let serverIP = window.location.host.toString();
    g.ws = new WebSocket(`ws://${serverIP}/ws/${g.userid}`);
    g.ws.onmessage = receive_ws;
    g.ws.onopen = function() {
        request_start();
    };
}

function send_ws(message)
{
    let g = Alpine.store('gamestate');
    let raw = JSON.stringify(message);
    let res = g.ws.send(raw);
}

function receive_ws(event) {
    // twice because string inside string
    let info = JSON.parse(JSON.parse(event.data));
    let g = Alpine.store('gamestate');
    let mainBtn = document.getElementById('main-button');
    let sideBtn = document.getElementById('side-button');
    if (info.data !== null) {
        console.log(info);
    }
    switch(info.type) {
        case "loading":
            logMessage(`${info.remain} players still need to connect`, 'is-secondary');
            g.statusz = "LOADING";
            addNotification('Waiting for other players to connect...', 'is-secondary');
            setButtonActivity(mainBtn, false);
            setButtonActivity(sideBtn, false);
            break;
        case "ready":
            g.statusz = "READY";
            setButtonActivity(mainBtn, true);
            setButtonActivity(sideBtn, false);
            break;
        case "log":
            // logMessage(JSON.stringify(info.data));
            processLog(info.data);
            g.history.push(info.data);
            break;
        case "select-attack":
            addNotification("Select cards for your attack.", 'is-primary');
            selectAttack(info.data);
            break;
        case "select-defend":
            addNotification("Select cards for your defense.", 'is-primary');
            selectDefend(info.data);
            break;
        case "select-redirect":
            logMessage(`select who plays next`, 'is-primary');
            selectRedirect(info.data);
            break;
        default:
            logMessage("fug", 'is-danger');
            logMessage(JSON.stringify(info));
            break;
    }
}

//
function reset_game() {
    // console.log("Resetting game");
    let g = Alpine.store('gamestate');
    document.getElementById('messages').replaceChildren();
    g.history = [];
    let message = {userid: g.userid, type:'player-reset', choice:0};
    send_ws(message);
}

function download_json() {
    let g = Alpine.store('gamestate');
    let text = JSON.stringify(g.history);
    let filename = `regi-${Date.now().toString()}.json`;
    let element = document.createElement('a');
    element.setAttribute('href', 'data:application/json;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);
    element.click();
}

function request_start() {
    let g = Alpine.store('gamestate');
    send_ws({userid: g.userid, username: g.username, type: "player-join", choice:0});
}
function player_ready() {
    let g = Alpine.store('gamestate');
    send_ws({userid: g.userid, type: "player-ready", choice:0});
}




function submit_option() {
    let g = Alpine.store("gamestate");
    if (g.redirection) {
        submit_redirect_option();
    } else {
        submit_card_option();
    }
}

function submit_redirect_option() {
    let g = Alpine.store("gamestate");
    let players = document.getElementById('other-players').firstChild;
    let pickset = new Set([]);
    const regex = /Player /i;
    for (let player of players.children) {
        if (player.classList.contains("is-focused")) {
            pickset.add(player.innerHTML.replace(regex, ""));
        }
    }
    console.log(pickset);

    let pickIndex = -1;
    let pickArr = Array.from(pickset);
    if (pickArr.length != 1) {
        logMessage(`Invalid redirection!`, 'is-warning');
        addNotification("Invalid Redirection! Pick Again", 'is-warning');
    } else {
        pickIndex = parseInt(pickArr[0]);
        // send the option picked
        let msg = { userid: g.userid, type: "player-move", choice: pickIndex };
        send_ws(msg);
        logMessage(`You selected Player ${Array.from(pickset)}`);
        // after sending your option, disable the buttons
        let submitter = document.getElementById('main-button');
        let yielder = document.getElementById('side-button');
        setButtonActivity(submitter, false);
        setButtonActivity(yielder, false);

        g.redirection = false;
    }
}

function submit_card_option() {
    let g = Alpine.store("gamestate");
    let cur_player = g.data;
    // get selected cards
    let cards = document.getElementById('player-cards').firstChild;
    let pickset = new Set([]);
    for (let card of cards.children) {
        if (card.classList.contains("is-focused")) {
            pickset.add(card.innerHTML);
        }
    }

    let picksOK = false;
    // console.log(cur_player);
    let comboLen = cur_player.combos.length;
    let pickIndex = -1;
    for (let i = 0; i < comboLen; i++) {
        let combset = new Set(cur_player.combos[i].map(x => x.value));
        if (pickset.isSubsetOf(combset) && combset.isSubsetOf(pickset)) {
            picksOK = true;
            pickIndex = i;
            break;
        }
    }

    if (!picksOK) {
        logMessage(`${Array.from(pickset)} is not a valid move!`, 'is-warning');
        updateCards(g.data.player);
        addNotification("Invalid Move! Pick Cards Again", 'is-warning');
    } else {
        // send the option picked
        let msg = { userid: g.userid, type: "player-move", choice: pickIndex };
        send_ws(msg);
        logMessage(`You selected ${Array.from(pickset)}`);
        // after sending your option, disable the buttons
        let submitter = document.getElementById('main-button');
        let yielder = document.getElementById('side-button');
        setButtonActivity(submitter, false);
        setButtonActivity(yielder, false);

    }
}

function yield_option() {
    let g = Alpine.store("gamestate");
    let cur_player = g.data;
    
    let picksOK = false;
    let comboLen = cur_player.combos.length;
    let pickIndex = -1;
    for (let i = 0; i < comboLen; i++) {
        if (combos[i].length == 0) {
            picksOK = true;
            pickIndex = i;
            break;
        }
    }
    
    if (!picksOK) {
        logMessage("yield is not a valid move!", 'is-warning');
        addNotification("Invalid Yield! Pick Cards", 'is-warning');
    } else {
        // send the option picked
        let msg = { userid: g.userid, choice: pickIndex };
        send_ws(msg);
        logMessage(`You selected ${Array.from(pickset)}`);
        // after sending your option, disable the buttons
        let submitter = document.getElementById('main-button');
        let yielder = document.getElementById('side-button');
        setButtonActivity(submitter, false);
        setButtonActivity(yielder, false);

    }
}

function setButtonActivity(button, active) {
    if (active) {
        button.classList.remove("is-loading")
    } else {
        button.classList.add("is-loading")
    }
}

function makeEnemyPileWithInfo(ce, enemyPileSize) {
    let wrapper = document.createElement("div");
    wrapper.style.display = "inline-block";
    wrapper.style.textAlign = "center";

    let depth = enemyPileSize > 0 ? Math.round((enemyPileSize / 12) * 8) : 0;

    let pile = document.createElement("div");
    pile.className = "box has-background-danger-light";
    pile.style.width = "90px";
    pile.style.height = "110px";
    pile.style.display = "flex";
    pile.style.flexDirection = "column";
    pile.style.alignItems = "center";
    pile.style.justifyContent = "center";
    pile.style.padding = "4px";
    pile.style.borderRadius = "8px";
    pile.style.border = "2px solid rgba(0,0,0,0.2)";
    pile.style.boxShadow = depth > 0
        ? `${depth}px ${depth}px 0px rgba(0,0,0,0.3), ${depth}px ${depth}px 0px -1px rgba(0,0,0,0.1)`
        : "none";
    pile.style.position = "relative";
    pile.style.marginBottom = depth + "px";

    let name = document.createElement("div");
    name.className = "has-text-weight-bold is-size-5";
    name.textContent = ce.value;
    pile.appendChild(name);

    let hp = document.createElement("div");
    hp.className = "is-size-6";
    hp.textContent = `${ce.hp} HP`;
    pile.appendChild(hp);

    let lbl = document.createElement("div");
    lbl.className = "is-size-7";
    lbl.textContent = `${enemyPileSize} enemies left`;
    lbl.style.marginTop = "4px";

    wrapper.appendChild(pile);
    wrapper.appendChild(lbl);
    return wrapper;
}

function makePileVisual(label, count, maxCount, colorClass) {
    let wrapper = document.createElement("div");
    wrapper.style.display = "inline-block";
    wrapper.style.textAlign = "center";
    wrapper.style.marginRight = "16px";
    wrapper.style.marginBottom = "8px";

    // Shadow depth based on card count (0-8px)
    let depth = maxCount > 0 ? Math.round((count / maxCount) * 8) : 0;

    let pile = document.createElement("div");
    pile.className = `box ${colorClass}`;
    pile.style.width = "70px";
    pile.style.height = "90px";
    pile.style.display = "flex";
    pile.style.alignItems = "center";
    pile.style.justifyContent = "center";
    pile.style.padding = "4px";
    pile.style.borderRadius = "8px";
    pile.style.border = "2px solid rgba(0,0,0,0.2)";
    pile.style.boxShadow = depth > 0
        ? `${depth}px ${depth}px 0px rgba(0,0,0,0.3), ${depth}px ${depth}px 0px -1px rgba(0,0,0,0.1)`
        : "none";
    pile.style.position = "relative";
    pile.style.marginBottom = depth + "px";

    let num = document.createElement("span");
    num.className = "is-size-3 has-text-weight-bold";
    num.textContent = count;
    pile.appendChild(num);

    let lbl = document.createElement("div");
    lbl.className = "is-size-7";
    lbl.textContent = label;
    lbl.style.marginTop = "4px";

    wrapper.appendChild(pile);
    wrapper.appendChild(lbl);
    return wrapper;
}

function makePilesRow(game) {
    let res = document.createElement("div");
    res.className = "mb-4";
    let hdr = document.createElement("h2");
    hdr.className = "subtitle is-5";
    hdr.textContent = "Piles";
    res.appendChild(hdr);

    let row = document.createElement("div");
    row.style.display = "flex";
    row.style.flexWrap = "wrap";
    let totalCards = game.draw_pile_size + game.discard_pile_size;
    row.appendChild(makePileVisual("Draw", game.draw_pile_size, totalCards || 42, "has-background-info-light"));
    row.appendChild(makePileVisual("Discard", game.discard_pile_size, totalCards || 42, "has-background-warning-light"));
    res.appendChild(row);
    return res;
}

function makeOtherPlayerInfo(game) {
    let res = document.createElement("div");
    res.className = "mb-4";
    let hdr = document.createElement("h2");
    hdr.className = "subtitle is-5";
    hdr.textContent = "Players";
    res.appendChild(hdr);

    let els = document.createElement("ul");
    for (let player of game.players) {
        let el = document.createElement("li");
        el.className = "is-size-6";
        el.appendChild(document.createTextNode(`${getDisplayName(player)}: ${player.num_cards} cards`));
        els.appendChild(el);
    }
    res.appendChild(els);
    return res;
}

function makeUsedCombos(combos) {
    let res = document.createElement("div");
    res.className = "mb-4";
    let hdr = document.createElement("h2");
    hdr.className = "subtitle is-5";
    hdr.textContent = "Combos Used";
    res.appendChild(hdr);

    let list = document.createElement("div");
    list.style.display = "flex";
    list.style.flexDirection = "column";
    list.style.alignItems = "center";
    for (let combo of combos) {
        let row = document.createElement("div");
        row.style.display = "flex";
        row.style.gap = "4px";
        row.style.marginBottom = "4px";
        row.style.justifyContent = "center";
        for (let card of combo) {
            let b = document.createElement("div");
            b.style.width = "36px";
            b.style.height = "48px";
            b.style.display = "flex";
            b.style.alignItems = "center";
            b.style.justifyContent = "center";
            b.style.borderRadius = "5px";
            b.style.border = "1px solid rgba(0,0,0,0.2)";
            b.style.backgroundColor = "#e8f0fe";
            b.style.boxShadow = "1px 1px 0px rgba(0,0,0,0.1)";
            b.className = "has-text-weight-bold is-size-7";
            b.textContent = card.value;
            row.appendChild(b);
        }
        list.appendChild(row);
    }
    res.appendChild(list);
    return res;
}

function updateBoard(game) {
    let game_view = document.getElementById('game-view');
    game_view.replaceChildren();
    game_view.appendChild(makePilesRow(game));
    game_view.appendChild(makeOtherPlayerInfo(game));

    let enemy_view = document.getElementById('enemy-view');
    enemy_view.replaceChildren();
    enemy_view.appendChild(makeEnemyPileWithInfo(game.current_enemy, game.enemy_pile_size));
    if (game.used_combos && game.used_combos.length > 0) {
        enemy_view.appendChild(makeUsedCombos(game.used_combos));
    }
}

function getCardButton(card, block) {
    let b = document.createElement("div");
    b.style.width = "50px";
    b.style.height = "68px";
    b.style.display = "flex";
    b.style.alignItems = "center";
    b.style.justifyContent = "center";
    b.style.borderRadius = "6px";
    b.style.border = "2px solid rgba(0,0,0,0.2)";
    b.style.cursor = "pointer";
    b.style.userSelect = "none";
    b.style.transition = "transform 0.15s, box-shadow 0.15s";
    b.style.backgroundColor = "#e8f0fe";
    b.style.color = "#363636";
    b.style.boxShadow = "2px 2px 0px rgba(0,0,0,0.15)";
    b.className = "has-text-weight-bold is-size-6";
    b.textContent = card;
    b.addEventListener("click", () => {
        if (b.classList.contains("is-focused")) {
            b.classList.remove("is-focused");
            b.style.backgroundColor = "#e8f0fe";
            b.style.color = "#363636";
            b.style.border = "2px solid rgba(0,0,0,0.2)";
            b.style.transform = "";
            b.style.boxShadow = "2px 2px 0px rgba(0,0,0,0.15)";
        } else {
            b.classList.add("is-focused");
            b.style.backgroundColor = "#363636";
            b.style.color = "white";
            b.style.border = "2px solid #3273dc";
            b.style.transform = "translateY(-8px)";
            b.style.boxShadow = "3px 5px 4px rgba(0,0,0,0.3)";
        }
    });
    return b;
}

function getPlayerButton(player, block) {
    let b = document.createElement("div");
    b.className = "button is-link";
    b.addEventListener("click", () => { 
        if (b.classList.contains("is-focused")) {
            b.classList.remove("is-focused") 
            b.classList.remove("is-dark")
            b.classList.add("is-link")
        } else {
            // mutually exclusive
            for (const otherb of block.children) {
                otherb.classList.remove("is-focused") 
                otherb.classList.remove("is-dark")
                otherb.classList.add("is-link")
            }
            b.classList.add("is-focused")
            b.classList.add("is-dark")
            b.classList.remove("is-link")
        }
    });
    b.innerHTML = getDisplayName(player);
    return b;
}

function updateCards(player) {
    let g = Alpine.store('gamestate');
    let target = document.getElementById('player-cards');
    target.replaceChildren();

    let bgroup = document.createElement("div");
    bgroup.style.display = "flex";
    bgroup.style.flexWrap = "wrap";
    bgroup.style.gap = "8px";
    let cards = player.cards;
    for (let i = 0, len = cards.length; i < len; i++) {
        bgroup.appendChild(getCardButton(cards[i]));
    }
    target.appendChild(bgroup);
}
function updateOtherPlayers(player, game) {
    let g = Alpine.store('gamestate');
    let target = document.getElementById('other-players');
    target.replaceChildren();

    let bgroup = document.createElement("div");
    let players = game.players;
    bgroup.className = "buttons has-addons are-medium";
    for (let i = 0, len = players.length; i < len; i++) {
        if (players[i].id != player.id) {
            bgroup.appendChild(getPlayerButton(players[i], bgroup));
        }
    }
    target.appendChild(bgroup);
    // console.log(player);
}

function selectAttack(data) {
    let g = Alpine.store('gamestate');
    g.data = data;
    g.redirection = false;
    let submitter = document.getElementById('main-button');
    let yielder = document.getElementById('side-button');
    setButtonActivity(submitter, true);
    setButtonActivity(yielder, data.yield_allowed);
    //
    g.myTurn = true;
    g.playerShould = "You have to ATTACK";
    // console.log(data);
    updateBoard(data.game);
    updateCards(data.player);
}
function selectDefend(data) {
    let g = Alpine.store('gamestate');
    g.data = data;
    g.redirection = false;
    let submitter = document.getElementById('main-button');
    let yielder = document.getElementById('side-button');
    setButtonActivity(submitter, true);
    setButtonActivity(yielder, false);
    //
    g.myTurn = true;
    g.playerShould = `You have to DEFEND ${data.damage} damage`;
    // console.log(data);
    updateBoard(data.game);
    updateCards(data.player);
}
function selectRedirect(data) {
    let g = Alpine.store('gamestate');
    g.data = data;
    g.redirection = true;
    let submitter = document.getElementById('main-button');
    let yielder = document.getElementById('side-button');
    setButtonActivity(submitter, true);
    setButtonActivity(yielder, false);
    //
    g.myTurn = true;
    g.playerShould = `You have to pick the next player`;
    addNotification("Select who will play next after you", 'is-primary');
    // console.log(data);
    updateBoard(data.game);
    updateOtherPlayers(data.player, data.game);
}

function getDisplayName(player) {
    return player.username || `Player ${player.id}`;
}

function updateCurrentPlayerCards(game) {
    let g = Alpine.store('gamestate');
    if (game && game.players) {
        let me = game.players.find(p => p.id == g.playerid);
        if (me && me.cards) {
            updateCards(me);
        }
    }
}

// logging
function processLog(data) {
    let g = Alpine.store('gamestate');
    if (data.game != null) {
        updateBoard(data.game);
        updateCurrentPlayerCards(data.game);
    }
    switch (data.event) {
        case 'STARTGAME':
            if (data.game != null && data.game.active_player_id !== null) {
                g.statusz = "RUNNING";
                logMessage("Game has started", 'is-primary');
            }
            break;
        case 'ATTACK':
            let combo1 = data.combo.map(x => x.value);
            logMessage(`${getDisplayName(data.player)} attacked ${data.enemy.value} with ${combo1}`, 'is-info');
            logMessage(`${getDisplayName(data.player)} dealt ${data.damage} damage!`, 'is-info');
            break;
        case 'DEFEND':
            let combo2 = data.combo.map(x => x.value);
            logMessage(`${data.enemy.value} attacked ${getDisplayName(data.player)} for ${data.damage} damage`, 'is-info');
            logMessage(`${getDisplayName(data.player)} blocked with ${combo2}`, 'is-info');
            break;
        case 'REDIRECT':
            let nextPlayerName = "Player " + data.next_playerid; // Fallback if username not available for next player
            if (data.game && data.game.players) {
                const nextPlayer = data.game.players.find(p => p.id === data.next_playerid);
                if (nextPlayer && nextPlayer.username) {
                    nextPlayerName = nextPlayer.username;
                }
            }
            logMessage(`${getDisplayName(data.player)} redirected play to ${nextPlayerName}`, 'is-info');
            break;
        case 'ENEMYKILL':
            let exact = data.enemy.hp === 0 ? " exact " : "";
            logMessage(`${data.enemy.value} killed` + exact + "!", 'is-success'); 
            break;
        case 'REPLENISH':
            logMessage(`${data.n_cards} cards added back to the draw pile`, 'is-info');
            break;
        case 'DRAWONE':
            logMessage(`${getDisplayName(data.player)} drew a card`);
            break;
        case 'DECKEMPTY':
            logMessage(`${getDisplayName(data.player)} cannot draw`);
            break;
        case 'ENDGAME':
            endGameStatusUpdate();
            logMessage("Game has ended", 'is-primary');
            break;
        case 'POSTGAME':
            endGameStatusUpdate();
            logMessage("Game over!", 'is-primary');
            // logMessage("Postgame: " + JSON.stringify(data.game));
            break;
        case 'FAILBLOCK':
            logMessage(`${data.enemy.value} attacked ${getDisplayName(data.player)} for ${data.damage}`, 'is-danger');
            logMessage(`${getDisplayName(data.player)} can block at most ${data.maxblock}!`, 'is-danger');
            break;
        case 'FULLBLOCK':
            logMessage(`${data.enemy.value} is blocked by ${getDisplayName(data.player)}`, 'is-info');
            break;
        case 'STATE':
            g.statusz = "RUNNING";
            if (data.game.active_player_id !== null) {
                if (data.game.active_player_id != g.playerid) {
                    g.myTurn = false;
                    let submitter = document.getElementById('main-button');
                    let yielder = document.getElementById('side-button');
                    setButtonActivity(submitter, false);
                    setButtonActivity(yielder, false);

                    let activePlayerName = "Player " + data.game.active_player_id;
                    if (data.game.active_player && data.game.active_player.username) {
                        activePlayerName = data.game.active_player.username;
                    }
                    logMessage(`Wait for ${activePlayerName} to play..`);
                } else {
                    updateCards(data.game.active_player);
                }
            }
        case 'DEBUG':
            // can I send something here?
            break;
        // default:
            // console.log(data.event);
    }
}

function setupTurnMessageHover() {
    let tray = document.getElementById('turn-message');
    if (tray.dataset.hoverSetup) return;
    tray.dataset.hoverSetup = 'true';

    tray.addEventListener('mouseenter', () => {
        tray.querySelectorAll('.notification').forEach(n => {
            clearTimeout(parseInt(n.dataset.fadeTimer));
            clearTimeout(parseInt(n.dataset.removeTimer));
            n.style.transition = '';
            n.style.opacity = '1';
        });
    });
    tray.addEventListener('mouseleave', () => {
        dismissOldNotifications(200);
    });
}

function restackNotifications() {
    // no-op: notifications are absolutely positioned via inline styles set on creation
}

function dismissOldNotifications(delay) {
    let tray = document.getElementById('turn-message');
    let notifications = tray.querySelectorAll('.notification');
    for (let i = 0; i < notifications.length; i++) {
        scheduleDismiss(notifications[i], delay + i * 300);
    }
}

function scheduleDismiss(el, delay) {
    let fadeTimer = setTimeout(() => {
        el.style.transition = 'opacity 3500ms ease-in-out';
        el.style.opacity = '0';
    }, delay);
    let removeTimer = setTimeout(() => {
        el.remove();
        restackNotifications();
    }, delay + 3600);
    el.dataset.fadeTimer = fadeTimer;
    el.dataset.removeTimer = removeTimer;
}

function addNotification(content, subtype) {
    let tray = document.getElementById('turn-message');
    setupTurnMessageHover();
    //
    let trayHeight = tray.offsetHeight;
    let inset = 12;
    let notifHeight = trayHeight - (inset * 2);

    let res = document.createElement('div');
    res.className = 'notification is-light is-size-5';
    res.classList.add(subtype);
    res.role = "alert";
    res.style.position = 'absolute';
    res.style.top = '10px';
    res.style.left = '12px';
    res.style.right = '8px';
    res.style.height = notifHeight + 'px';
    res.style.maxHeight = notifHeight + 'px';
    res.style.margin = '0';
    res.style.padding = '8px 30px 8px 12px';

    let delButton = document.createElement('button');
    delButton.classList.add('delete');
    delButton.addEventListener("click", () => {
        res.remove();
        restackNotifications();
    });

    res.appendChild(delButton);
    res.appendChild(document.createTextNode(content));

    // Ensure newest is always on top
    let existing = tray.querySelectorAll('.notification');
    existing.forEach(n => { n.style.zIndex = '1'; });
    res.style.zIndex = '10';
    res.style.animation = 'slideInFromLeft 400ms ease-out forwards';
    tray.appendChild(res);
    dismissOldNotifications(600);
}

function logMessage(content, subtype=null) {
    let g = Alpine.store('gamestate');
    // add to log at the bottom
    let messages = document.getElementById('messages')
    let message = document.createElement('li')
    message.appendChild(document.createTextNode(content))
    messages.appendChild(message)
    if (subtype != null) {
        setTimeout(() => {addNotification(content, subtype);}, 1000);
    }
}

// UI
function gameStatusShowable() {
    let g = Alpine.store('gamestate');
    let showStates = ["READY", "RUNNING", "ENDED"];
    return showStates.includes(g.statusz);
}

function endGameStatusUpdate() {
    let g = Alpine.store('gamestate');
    g.statusz = "ENDED";
    let submitter = document.getElementById('main-button');
    let yielder = document.getElementById('side-button');
    setButtonActivity(submitter, true);
    setButtonActivity(yielder, true);
}

function mainButtonText() {
    let g = Alpine.store('gamestate');
    let result = "";
    switch (g.statusz) {
        case "LOADING":
            result = "Waiting...";
            break;
        case "READY":
            result = "Ready!"
            break;
        case "RUNNING":
            result = "Submit";
            break;
        case "ENDED":
        case "ERROR":
            result = "Restart";
            break;
        default:
            result = "???";
    }
    return result;
}
function mainButtonRedirect() {
    let button = document.getElementById("main-button");
    if (button.classList.contains("is-loading")) {
        return;
    }
    let g = Alpine.store('gamestate');
    switch (g.statusz) {
        case "LOADING":
            break;
        case "READY":
            player_ready();
            break;
        case "RUNNING":
            submit_option();
            break;
        case "ENDED":
        case "ERROR":
            reset_game();
            break;
    }
}

function sideButtonText() {
    let g = Alpine.store('gamestate');
    let result = "";
    switch (g.statusz) {
        case "LOADING":
        case "READY":
        case "ERROR":
            result = "<hide>";
            break;
        case "RUNNING":
            result = "Yield";
            break;
        case "ENDED":
            result = "Download JSON";
            break;
        default:
            result = "???";
    }
    return result;
}

function sideButtonRedirect() {
    let g = Alpine.store('gamestate');
    let button = document.getElementById("side-button");
    if (button.classList.contains("is-loading")) {
        return;
    }
    switch (g.statusz) {
        case "LOADING":
            break;
        case "READY":
            break;
        case "RUNNING":
            yield_option();
            break;
        case "ENDED":
        case "ERROR":
            download_json();
            break;
    }
}



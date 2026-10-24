// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/ERC20.sol";
import "@openzeppelin/contracts/token/ERC20/extensions/ERC20Burnable.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title VeriForestToken ($VCO2)
 * @notice Verifiable real-world carbon credits backed by satellite AI MRV.
 * 1 Token = 1 Metric Tonne of verified CO2 sequestered.
 */
contract VeriForestToken is ERC20, ERC20Burnable, Ownable {

    address public oracleBackend; // The Python AI Oracle address authorized to mint

    struct VerificationProof {
        uint256 timestamp;
        uint256 tonnes;
        string satelliteProofHash;
    }

    mapping(address => VerificationProof[]) public projectProofs;

    // Events for radical on-chain transparency
    event CarbonCreditsMinted(address indexed forestOwner, uint256 amount, string proofHash);
    event CarbonRetired(address indexed burner, uint256 amount, string companyName, string reason);

    modifier onlyOracle() {
        require(msg.sender == oracleBackend || msg.sender == owner(), "Caller is not authorized AI oracle");
        _;
    }

    constructor(address _oracleBackend) ERC20("VeriForest Carbon Credit", "VCO2") Ownable(msg.sender) {
        oracleBackend = _oracleBackend;
    }

    function setOracleBackend(address _newOracle) external onlyOwner {
        oracleBackend = _newOracle;
    }

    /**
     * @notice Mints new tokens only when the AI Satellite Oracle approves genuine biomass growth.
     */
    function mintCredits(address forestOwner, uint256 tonnes, string calldata satelliteProofHash) external onlyOracle {
        _mint(forestOwner, tonnes);
        
        projectProofs[forestOwner].push(VerificationProof({
            timestamp: block.timestamp,
            tonnes: tonnes,
            satelliteProofHash: satelliteProofHash
        }));

        emit CarbonCreditsMinted(forestOwner, tonnes, satelliteProofHash);
    }

    /**
     * @notice Emitting companies burn/retire credits to claim Net-Zero.
     * Prevents double-counting permanently.
     */
    function retireCredits(uint256 tonnes, string calldata companyName, string calldata reason) external {
        burn(tonnes); // Permanently destroys tokens
        emit CarbonRetired(msg.sender, tonnes, companyName, reason);
    }
}